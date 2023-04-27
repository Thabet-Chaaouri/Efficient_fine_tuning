# Efficient_fine_tuning

## Problem

LLM are big Models with billions of parameters, For example:

Take the 7B Llama model.
A typical model trained in mixed precision with AdamW requires 18 bytes per model parameter plus activation memory:
- 4 bytes * number of parameters for fp32 training (in bf16, every parameter uses 2 bytes)
- 8 bytes * number of parameters for normal AdamW (maintains 2 states)
- 4 bytes * number of parameters for gradients
- About 2 bytes for Activations

So a 7B parameter model would use (2+2)7B=28GB just to fit in memory (if we use bf16 and 8-bit AdamW) and would likely need more when you compute intermediate values such as attention scores.

So you’ll run out of memory sooner or later.

## Solution

Parameter-Efficient Fine-Tuning (PEFT) techniques
