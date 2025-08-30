#!/usr/bin/env python3
"""
Train a single special token <EMPTY2> embedding such that its positions predict the next-next token.
Frozen decoder-only LM, only the new token's embedding is trained.

Usage example:
python train_empty2.py \
  --model_name_or_path Qwen/Qwen3-14B \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --split train \
  --output_dir ./out_empty2 \
  --per_device_batch_size 8 \
  --learning_rate 5e-3 \
  --max_steps 2000 \
  --logging_steps 10 \
  --save_steps 500 \
  --max_seq_len 128 \
  --mask_perc 0.2 \
  --max_slots 64 \
  --seed 42 \
  --wandb_project empty_tokens
"""

import argparse
import random
import os
import math
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    logging as hf_logging,
)
from datasets import load_dataset
import wandb
from tqdm.auto import tqdm

hf_logging.set_verbosity_error()  # reduce HF chatter

def save_empty2_embedding(model, tokenizer, output_dir, fname = f"empty2_embedding.pt"):
    empty_id = tokenizer.convert_tokens_to_ids("<EMPTY2>")
    vec = model.get_input_embeddings().weight[empty_id].detach().cpu()
    path = os.path.join(output_dir, fname)
    torch.save(vec, path)
    print(f"Saved <EMPTY2> embedding to {path}")
    return path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_file", type=str, default=None, help="local file (.txt or jsonl) to load instead of dataset_name")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--mask_perc", type=int, default=0.2, help="What percent of the sequence to mask")
    p.add_argument("--max_slots", type=int, default=64, help="max number of <EMPTY2> slots to place after prefix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_false", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=True, help="use bitsandbytes 8-bit loading (optional)")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_tokenizer_and_model(model_name: str, tokenizer: AutoTokenizer, args):
    # Add special token <EMPTY2> if not present
    special = "<EMPTY2>"
    if special not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [special]})
    empty_id = tokenizer.convert_tokens_to_ids(special)

    # Load model
    # Optionally load in 8-bit (requires bitsandbytes & transformers support)
    model_kwargs = {"torch_dtype": torch.float16} if args.fp16 else {}
    if args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        except Exception as e:
            print("Failed to load in 8-bit. Install bitsandbytes and use compatible transformers.")
            raise e
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(args.device)

    # resize token embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only the embedding vector(s) corresponding to <EMPTY2>
    # Depending on model architecture, embeddings are in model.get_input_embeddings().weight
    input_emb = model.get_input_embeddings()
    # Ensure parameters of embedding remain requiring grad only for the new token row
    # Approach: create an embedding parameter that we update instead of whole matrix.
    # Simpler: set entire embedding to requires_grad=False, then make the row a separate nn.Parameter and replace row in forward via hook.
    # But easiest: set requires_grad True on the embedding weight, and register a mask to zero grads for all rows except empty_id after backward.
    # We'll use grad hook to zero-out grads for non-empty_id rows.

    # Enable grads for embedding; we'll mask during optimizer step
    input_emb.weight.requires_grad = True

    return model, tokenizer, empty_id


def collate_fn_batch(examples: List[Dict], tokenizer, empty_id, args):
    """examples: list of dicts with keys 'input_ids' and 'labels' (list[int])"""
    # Convert to tensors
    batch_input_ids = [e["input_ids"] for e in examples]
    batch_labels = [e["labels"] for e in examples]

    # Find max length for this batch (capped at max_seq_len)
    max_len = max(len(x) for x in batch_input_ids)
    max_len = min(max_len, args.max_seq_len)

    padded_inputs, padded_labels, attention_masks = [], [], []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for ids, lbls in zip(batch_input_ids, batch_labels):
        # Truncate
        ids = ids[:max_len]
        lbls = lbls[:max_len]

        pad_len = max_len - ids.size(0)

        if pad_len > 0:
            ids = torch.cat([
                ids,
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
            lbls = torch.cat([
                lbls,
                torch.full((pad_len,), -100, dtype=torch.long)   # <- pad labels with -100
            ])

        padded_inputs.append(ids)
        padded_labels.append(lbls)
        attention_masks.append((ids != pad_token_id).long())

    input_ids = torch.stack(padded_inputs, dim=0)
    labels = torch.stack(padded_labels, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def make_examples_from_text(example_text: str, tokenizer, args, strategy='prefix') -> List[Dict]:
    """
    Given a raw text string, produce a single example where:
      - we keep 1-args.mask_perc tokens of the original text (if possible),
      - then we replace the next tokens with <EMPTY2> up to args.max_slots or end-of-text.
    The example stores tokenized input_ids for the model.
    """
    # tokenization (no truncation here; we'll truncate later)
    tok = tokenizer(example_text, add_special_tokens=False, return_tensors="pt")
    ids_tensor = tok["input_ids"].squeeze(0)  # remove batch dim
    if ids_tensor.numel() == 0:
        return []

    # build labels shifted by +2
    labels = ids_tensor.clone()
    if labels.numel() > 2:
        labels[:-2] = ids_tensor[2:]
        labels[-2:] = -100  # ignore last two tokens
    else:
        labels[:] = -100
    
    
    # Calculate how many tokens to mask
    total_tokens = ids_tensor.numel()
    mask_tokens = int(total_tokens * args.mask_perc)
    
    # Limit masking to args.max_slots if specified
    mask_tokens = min(mask_tokens, args.max_slots)
    # Create modified input_ids
    empty_token_id = tokenizer.convert_tokens_to_ids("<EMPTY2>")
    if mask_tokens > 0:
        if strategy == 'random':
            mask_positions = torch.randperm(total_tokens)[:mask_tokens]
            ids_tensor[mask_positions] = empty_token_id
        else:
            ids_tensor[mask_tokens:] = empty_token_id

    # prefix tokens
    # prefix_len = min(args.prefix_tokens, ids_tensor.size(0))
    # prefix = ids_tensor[:prefix_len].tolist()
    # rest_len = ids_tensor.size(0) - prefix_len

    # # number of slots to place
    # num_slots = min(rest_len, args.max_slots)
    # if prefix_len + num_slots > args.max_seq_len:
    #     num_slots = max(0, args.max_seq_len - prefix_len)

    # # construct input_ids: prefix + <EMPTY2> slots
    # empty_token_id = tokenizer.convert_tokens_to_ids("<EMPTY2>")
    # if empty_token_id is None:
    #     raise ValueError("<EMPTY2> not found in tokenizer vocab")

    # input_ids = torch.tensor(prefix + [empty_token_id] * num_slots, dtype=torch.long)

    # # truncate labels to match input_ids length (optional but recommended)
    # labels = labels[:input_ids.size(0)]
    return [{"input_ids": ids_tensor, 'labels' : labels}]


# def dataset_to_examples(dataset, tokenizer, args, text_key="text"):
#     """Stream dataset and yield examples; small wrapper used by DataLoader creation."""
#     examples = []
#     for item in dataset:
#         text = item.get(text_key) if isinstance(item, dict) else item
#         exs = make_examples_from_text(text, tokenizer, args)
#         examples.extend(exs)
#     return examples


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # init wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # load tokenizer; keep fast tokenizer if available
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # ensure pad token exists (some decoder models don't)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Prepare model + tokenizer + empty_id
    model, tokenizer, empty_id = prepare_tokenizer_and_model(args.model_name_or_path, tokenizer, args)
    breakpoint()

    # Freeze everything except embeddings (we'll mask grads after backward)
    for name, p in model.named_parameters():
        p.requires_grad = False
    # But embeddings weight we enabled earlier in prepare... ensure it's True
    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True

    # We'll ensure grads are zero'ed for all rows except empty_id after backward by masking in optimizer step

    # Load dataset
    if args.dataset_file:
        # load local text lines
        raw_ds = load_dataset("text", data_files={"train": args.dataset_file})
        dataset_split = raw_ds[args.split]
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config)  # may have multiple splits
        dataset_split = dataset[args.split]
    else:
        raise ValueError("Please specify --dataset_name or --dataset_file")
    # Turn raw text dataset into examples list (this is memory-limited for very large corpora)
    # For large corpora you should build a streaming DataLoader; for quick experiments we collect examples.
    print("Building examples (this may take a bit)...")
    raw_text_key = "text" if "text" in dataset_split.column_names else dataset_split.column_names[0]
    examples = []
    # streaming large dataset in batches
    for page in dataset_split:
        txt = page.get(raw_text_key) if isinstance(page, dict) else page
        exs = make_examples_from_text(txt, tokenizer, args)
        if exs:
            examples.extend(exs)
        # to avoid using too much memory, optionally break early in debug
        # (user controls max_steps)
        if len(examples) > 200000:
            break

    if len(examples) == 0:
        raise ValueError("No examples were created from dataset. Check tokenization / prefix_tokens / max_slots.")

    # DataLoader + collate
    dataloader = DataLoader(
        examples,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_batch(batch, tokenizer, empty_id, args),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer: only embedding weights (but embedding weight includes full vocab; we will mask grads later)
    optimizer = torch.optim.AdamW([p for p in model.get_input_embeddings().parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)

    # Scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.load_in_8bit:
        model.to(device)
    model.train()

    #scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Loss (we will compute CE only on positions where labels != -100)
    ce_loss = CrossEntropyLoss(ignore_index=-100, reduction="mean")

    global_step = 0
    running_loss = 0.0
    pbar = tqdm(total=args.max_steps, desc="training")
    dataloader_iter = iter(dataloader)

    # Precompute mask tensor index for empty token to speed up selection logic
    empty_token_id = empty_id

    
    while global_step < args.max_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)


        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, S, V)
            
            # We want to compute CE only on positions where input_id == empty_token_id,
            # with targets = labels (which we constructed as the token at pos+2 or -100)
            # Extract logits at empty positions: reshape to (-1, V) and filter by mask
            #mask = (input_ids == empty_token_id)  # (B, S)
            mask = (input_ids == empty_token_id) & (labels != -100)
            if mask.sum() == 0:
                # nothing to train on this batch; skip
                continue
            masked_logits = logits[mask]  # (#empty_positions, V)
            masked_labels = labels[mask]   # (#empty_positions,)
            if global_step == 0:
                # debug
                print(f'Labels: {labels[0]}')
                print(f'Inputs: {input_ids[0]}')
                
            # masked_labels may contain -100; CrossEntropyLoss handles ignore_index
            loss = ce_loss(masked_logits, masked_labels)

            # shift_logits = logits[:, 1:, :]        # prediction for positions 1..S (based on inputs 0..S-1)
            # shift_labels = labels[:, 1:]           # align labels
            # prev_is_empty = (input_ids[:, :-1] == empty_token_id)

            # mask = prev_is_empty & (shift_labels != -100)
            # if mask.any():
            #     loss = ce_loss(shift_logits[mask], shift_labels[mask])
            # else:
            #     continue

        loss.backward()

        # Before stepping, zero gradients for embedding rows except empty_token_id
        # embedding grads shape: (vocab_size, emb_dim)
        emb_grad = model.get_input_embeddings().weight.grad

        if emb_grad is not None:
            with torch.no_grad():
                # Create a mask to zero all gradients except for empty_token_id
                grad_mask = torch.zeros_like(emb_grad)
                grad_mask[empty_token_id] = 1.0
                emb_grad.mul_(grad_mask)
    
        # unscale & step
        optimizer.step()
        lr_scheduler.step()

        loss_item = loss.detach().item()
        running_loss += loss_item
        global_step += 1
        pbar.update(1)

        if args.wandb_project:
            wandb.log({"loss": loss_item, "lr": lr_scheduler.get_last_lr()[0], "step": global_step})

        if global_step % args.logging_steps == 0:
            pbar.set_postfix({"loss": f"{(running_loss/args.logging_steps):.4f}"})
            running_loss = 0.0

        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            # Save model + tokenizer; but only embedding changed, so saving full model is fine (small overhead)
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)
            # save only embedding well? Simpler: save full model & tokenizer
            #model.save_pretrained(save_path)
            #tokenizer.save_pretrained(save_path)

            save_empty2_embedding(model, tokenizer, save_path)

    pbar.close()

    # final save
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    save_empty2_embedding(model, tokenizer, args.output_dir, step=None)


    print("Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()
