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

#### Mixed precision :
You can do full finetuning in mixed precision with fp16 or bf16.
FP16 has limitations: reduced precision can lead to inaccurate weight updates, gradient underflows, and activation/loss overflows. To address these, mixed precision training is used:
- Forward and backward passes are done in FP16.
- Weight updates are done in FP32 using a master copy of the model. (so you can't load the model in fp16 for full finetuning, it sould be loaded in full precision)
- Gradient scaling is introduced to prevent underflows by multiplying the loss before backpropagation, then rescaling gradients after.

BF16 (BFloat16), another 16-bit format with the same range as FP32 but lower precision. BF16 avoids gradient scaling and is supported on newer hardware (Ampere+). It is recommended to work with bf16 as it is more stable.

#### PEFT techniques :
- Loading in 8bit/4bit precision
- Usinng Lora adapters to do efficient fine-tuning 

One way is using QLora : 
- leveraging NF4 (normalized float 4 (default)) or FP4 storage dtype. NF4 has shown to achieve better performance.
- using a second quantization after the first one to save an additional 0.4 bits per parameter.
- Chosing a proper computation dtype (float16, bfloat16, float32 etc), because we can't perform computation in 4 bits

```
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

------------------------------------------------------------

Parameter-Efficient Fine-Tuning (PEFT) techniques:
- Prefix Tuning
- Adapters
- LLama-Adapter method : a combination of prefix tuning and adapter method with some changes checkout [the LLama-adapter implementation](https://github.com/ZrrSkywalker/LLaMA-Adapter)
- LoRa

In the TRL-PEFT blog post, fine tuning 20B model took a single 24GB GPU thank to these techinques:
- Load the model in 8-bit precision : Quantize the model using the HF transformer library by precising load_in_8bit=True argument when calling .from_pretrained method
- Add extra trainable adapters using peft and reduce the memory requirements for the optimizer states  by training only the adapters parameters.
- get a reference model and an active logits: with peft ther is no need to copy the model, using the disable_adapters parameter, the library uses the same model and reduce memory

In Sebastien Blog Post, a comparison between Basic, LoRA and Adapter finetuning for lit-LLama from Lightning on A100, the same batch size gives : 
- Adapter : 
    * used about 22 Gb
    * finished 62,400 iterations in 162 min
 - LoRA :
    * used 21 Gb of memory
    * finished in 192 min. In sum, Adapter and LoRA use approximately the same amount of RAM and have roughly the same training time based on the Lit-LLaMA implementations.
  - full finetuning : 
    * required at least 2 GPUs with at least 30 Gb
    * Fully sharded training to distribute the weights. 
    * Or 4 GPUs with a maximum memory usage of 22 Gb per GPU. The training on 4 GPUs took 1956 min. This would be at least 6,000 min on a single GPU, which would be 30-40x more expensive than the parameter-efficient LLaMA-Adapter or LoRA alternatives.

A comparatif fine tuning is done using the prefix and basic finetuning in a [google colab notebook](https://github.com/Thabet-Chaaouri/Efficient_fine_tuning/blob/main/Prefix_VS_Basic_fine_tuning.ipynb) on T5 large model, to do sentiment analysis:
- with basic tuning, we can't fit the model in A100 GPU and had to use accelerator and mixed precision fp16
- With prefix tuning, the model did fit easily.

A comparatif fine tuning is done between prompt and full finetuning in a [Google Colab notebook](https://github.com/Thabet-Chaaouri/Efficient_fine_tuning/blob/main/Prompt_VS_Basic_Tuning.ipynb) on Bloomz 560M model to calssify complaint text:
- epoch duration = 1mn40s with prompt tuning
- epoch duration = -- with full finetuning

An example of finetuning is done using 8 bit precision on OPT 6.7B parameters:
- Without quantization the model would need about 14Gb just to load the model with fp16 mixed precision and 28Gb in full precision
- With quantization it took only 7 Gb to load the model then with LoRA it was possible to train it on one single GPU in [Kaggle Notebook](https://www.kaggle.com/code/thabetchaaouri/notebook771577603e/edit)

## Tools
PEFT HuggingFace library : https://github.com/huggingface/peft

## Materials
Basic Finetuning LLM : 
- [x] Better Call Bloom : https://pub.towardsai.net/training-a-large-language-model-to-give-non-legal-advice-b9f6d7d11016 
- [ ] Basic Causal language model training : https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt 

Efficient finetuning techniques materials:
- [x] TRL-PEFT Blog post: https://huggingface.co/blog/trl-peft
- [x] Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters : https://lightning.ai/pages/community/article/understanding-llama-adapters/
- [x] Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA) : https://lightning.ai/pages/community/tutorial/lora-llm/ + [code](https://github.com/rasbt/low-rank-adaptation-blog)
- [x] PEFT HuggingFace Library : https://huggingface.co/docs/peft/index

Efficient finetuning examples:
- [x] Finetuning LLama with TRL and PEFT : https://huggingface.co/blog/stackllama
- [x] HF PEFT examples : https://huggingface.co/docs/peft/index & Github : https://github.com/huggingface/peft
- [ ] Finetuning Lit-LLama with Lora and Adapters : https://github.com/Lightning-AI/lit-llama + code : https://github.com/rasbt/low-rank-adaptation-blog
- [ ] Finetuning Bloom : https://www.philschmid.de/bloom-sagemaker-peft
- [ ] LLama-adapter : https://github.com/ZrrSkywalker/LLaMA-Adapter

## Datasets
- Alpaca dataset : https://github.com/tatsu-lab/stanford_alpaca
- Databricks Dolly dataset : https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Checkout the [self-instruct paper](https://arxiv.org/abs/2212.10560), the method used to generate the Alpaca dataset 

