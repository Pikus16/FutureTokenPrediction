"""
Next-Next Token Prediction Finetuning Pipeline
Finetunes only the output head of a pretrained LLM for next-next token prediction
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from torch.utils.checkpoint import checkpoint
import glob

@dataclass
class TrainingArguments:
    """Training configuration"""
    model_name: str = "Qwen/Qwen3-4B"
    dataset_name: str = 'BAAI/Infinity-Instruct'
    dataset_config: Optional[str] = '7M'
    output_dir: str = "./infinity_next_next_token_model"
    
    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 2
    warmup_steps: int = 500
    max_steps: int = -1
    save_freq: int = 1
    
    # Optimization
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    compile_model: bool = False  # PyTorch 2.0+ compilation
    
    # Data processing
    max_length: int = 512
    num_workers: int = 8
    preprocessing_num_workers: int = 16
    
    # Logging
    logging_steps: int = 10
    #eval_steps: int = 500
    save_steps: int = 5000
    wandb_project: str = "empty_tokens"
    wandb_entity: Optional[str] = None
    
    # Misc
    seed: int = 42


class NextNextTokenDataset(Dataset):
    """Dataset for next-next token prediction"""
    
    def __init__(self, tokenized_texts, max_length=512):
        self.examples = []
        for tokens in tokenized_texts:
            if len(tokens['input_ids']) > 2:  # Need at least 3 tokens
                self.examples.append(tokens['input_ids'])
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx][:self.max_length]
        
        # Create input_ids
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create labels for next-next token prediction
        # Shift by 2 positions and pad with -100
        if len(tokens) > 2:
            labels = tokens[2:] + [-100, -100]
        else:
            labels = [-100] * len(tokens)
        
        labels = torch.tensor(labels[:len(tokens)], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


class NextNextTokenModel(nn.Module):
    """Wrapper model for next-next token prediction with frozen base model"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Freeze all parameters except the output head
        lm_head_set = False
        for name, param in self.base_model.named_parameters():
            if 'lm_head' not in name:
                param.requires_grad = False
            if name == 'lm_head':
                lm_head_set = True

        if not lm_head_set:
            # Tied
            hidden_size = self.base_model.model.embed_tokens.embedding_dim  # 2560 for Qwen3-4B
            vocab_size = self.base_model.lm_head.out_features               # 151936

            # New trainable head
            new_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            # Optionally initialize from the original tied head
            with torch.no_grad():
                new_lm_head.weight.copy_(self.base_model.lm_head.weight)
            new_lm_head.requires_grad = True
            self.base_model.lm_head = new_lm_head

        # Debug
        print('Training below parameters:')
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                print(f'\t{name}')
        trainable_params = round(sum(p.numel() for p in self.base_model.parameters() if p.requires_grad) / (10 ** 6), 2)
        print(f'{trainable_params} million trainable params')
        
        # Create new trainable output head
        # hidden_size = base_model.lm_head.in_features
        # vocab_size = base_model.lm_head.out_features
        # self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # with torch.no_grad():
        #     self.lm_head.weight.copy_(base_model.lm_head.weight)
        
        # # Make only lm_head trainable
        # self.lm_head.weight.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get hidden states from frozen base model
        # with torch.no_grad():
        #     outputs = self.base_model.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True
        #     )
        #     hidden_states = outputs.last_hidden_state
        
        # # Apply trainable head
        # logits = self.lm_head(hidden_states)
        logits = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
        
        loss = None
        if labels is not None:
            # Flatten the tokens
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()
            # Calculate loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits}


def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

def _map_conversations_to_chat_template(example, tokenizer, add_generation_prompt=False):
    """
    Maps the BAAI/Infinity-Instruct conversation format to work with tokenizer.apply_chat_template.
    
    Args:
        example: A single example from the dataset containing 'conversations' field
        tokenizer: The tokenizer with chat template support
        add_generation_prompt: Whether to add generation prompt for training
    
    Returns:
        dict: Mapped example with 'text' field containing the formatted conversation
    """
    conversations = example['conversations']
    
    # Map the conversation format
    messages = []
    for turn in conversations:
        # Map 'human' to 'user' and 'gpt' to 'assistant'
        if turn['from'] == 'human':
            role = 'user'
        elif turn['from'] == 'gpt':
            role = 'assistant'
        elif turn['from'] == 'system':
            role = 'system'
        else:
            # Handle any other roles by keeping them as is
            raise ValueError(f'Unknown role: {role}')
        
        messages.append({
            'role': role,
            'content': turn['value']
        })
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )
    return {'text': formatted_text}

def load_and_prepare_dataset(args, tokenizer):
    """Load and tokenize the dataset"""
    print(f"Loading dataset: {args.dataset_name}")
    
    sampling_size = 200000
    if args.dataset_name == "rojagtap/bookcorpus":
        dataset = load_dataset("rojagtap/bookcorpus", split=f"train[:{sampling_size}]")  # Sample for speed
        text_column = "text"
    elif args.dataset_name == 'BAAI/Infinity-Instruct':
        dataset = load_dataset(args.dataset_name, args.dataset_config, split='train')
        mapped_dataset = dataset.map(
            lambda x: _map_conversations_to_chat_template(x, tokenizer, add_generation_prompt=False),
            num_proc=64,
            desc="Formatting conversations",
            remove_columns=dataset.column_names  # Remove all original columns
        )
        text_column='text'
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=f"train[:{sampling_size}]")
        # Assume first text column
        text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset.column_names
    )
    
    return NextNextTokenDataset(tokenized_dataset, args.max_length)


def train_epoch(model, dataloader, optimizer, scheduler, accelerator, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        with accelerator.autocast():
            outputs = model(**batch)
            loss = outputs['loss']
            loss = loss / args.gradient_accumulation_steps
        
        accelerator.backward(loss)
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Logging
        if step % args.logging_steps == 0:
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            if accelerator.is_main_process:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/step': step
                })
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, accelerator):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs['loss']
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Next-Next Token Finetuning")
    
    # Add all arguments from TrainingArguments
    for field_name, field_def in TrainingArguments.__dataclass_fields__.items():
        field_type = field_def.type
        default_value = field_def.default
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is type(Optional):
            field_type = field_type.__args__[0]
        
        # Convert type for argparse
        if field_type == bool:
            parser.add_argument(f'--{field_name}', type=lambda x: x.lower() == 'true', 
                              default=default_value)
        elif field_type == int:
            parser.add_argument(f'--{field_name}', type=int, default=default_value)
        elif field_type == float:
            parser.add_argument(f'--{field_name}', type=float, default=default_value)
        else:
            parser.add_argument(f'--{field_name}', type=str, default=default_value)
    
    args = parser.parse_args()
    args = TrainingArguments(**vars(args))
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize W&B
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"next_next_{args.model_name.split('/')[-1]}"
        )
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # if args.mixed_precision == 'fp16':
    #     precision = torch.float16
    # elif args.mixed_precision == 'bf16':
    #     precision = torch.bfloat16
    # else:
    #     precision = torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # torch_dtype=precision,
        device_map="auto"
    )
    
    # Enable gradient checkpointing on base model if requested
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    
    # Create the next-next token model
    model = NextNextTokenModel(
        base_model=base_model,
    )
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Load and prepare dataset
    train_dataset = load_and_prepare_dataset(args, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Split for eval (10% of training data)
    eval_size = len(train_dataset) // 10
    eval_dataset = torch.utils.data.Subset(train_dataset, range(eval_size))
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for eval
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.base_model.lm_head.parameters(),  # Only optimize the head
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    # Training loop
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "epoch*.pt")))
    start_epoch, best_loss = 0, float('inf')
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        print(f"Resuming from checkpoint: {latest_ckpt}")
        #ckpt = torch.load(latest_ckpt, map_location="cpu")
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)

        # Load lm_head only
        model.base_model.lm_head.load_state_dict(ckpt['lm_head_state_dict'])

        # Optimizer + scheduler states
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # Training state
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best loss so far {best_loss:.4f}")
    else:
        print("No checkpoint found, starting fresh training.")

    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, accelerator, args
        )
        
        # Evaluate
        eval_loss = evaluate(model, eval_dataloader, accelerator)
        
        print(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        
        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'eval/loss': eval_loss
            })
            
            # Save best model
            if eval_loss < best_loss:
                best_loss = eval_loss
                print(f"New best model with loss: {best_loss:.4f}")
                
            if epoch % args.save_steps == 0:
                # Save only the lm_head
                torch.save({
                    'lm_head_state_dict': model.base_model.lm_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'curr_loss':eval_loss,
                    'args': args
                }, os.path.join(args.output_dir, f'epoch{epoch}.pt'))
    
    # Final save
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'lm_head_state_dict': model.base_model.lm_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': args.num_epochs,
            'final_loss': eval_loss,
            'args': args
        }, os.path.join(args.output_dir, 'final_model.pt'))
        
        print(f"Training completed! Final eval loss: {eval_loss:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()
