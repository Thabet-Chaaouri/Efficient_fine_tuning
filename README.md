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

One real case, Check out Better Call Bloom : in this example Thomas finetuned without efficient techniques 3B Bloom for causal momdeling on A100 40GB GPU for 3 epochs, and it took him 26 hours. Joe Papa in his workshop showed that only using Pytorch 2.0 took down the duration to 16 hours.

## Solution

Parameter-Efficient Fine-Tuning (PEFT) techniques

In the TRL-PEFT blog post, fine tuning 20B model took a single 24GB GPU thank to these techinques:
- Load the model in 8-bit precision : Quantize the model using the HF transformer library by precising load_in_8bit=True argument when calling .from_pretrained method
- Add extra trainable adapters using peft and reduce the memory requirements for the optimizer states  by training only the adapters parameters.
- get a reference model and an active logits: with peft ther is no need to copy the model, using the disable_adapters parameter, the library uses the same model and reduce memory
- 



## Tools
PEFT HuggingFace library : https://github.com/huggingface/peft

## Materials
Basic Finetuning LLM : 
- [x] Better Call Bloom : https://pub.towardsai.net/training-a-large-language-model-to-give-non-legal-advice-b9f6d7d11016 
- [] Basic Causal language model training : https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt 

Efficient finetuning techniques materials:
- [x] TRL-PEFT Blog post: https://huggingface.co/blog/trl-peft

Efficient finetuning examples:
- [x] Finetuning LLama with TRL and PEFT : https://huggingface.co/blog/stackllama 
- [] Finetuning Bloom : https://www.philschmid.de/bloom-sagemaker-peft
- 

