### Fine-Tuning of LLAMA2 Using QLORA

---

## Introduction

In recent years, large language models (LLMs) have exhibited remarkable capabilities across various tasks. However, their increasing size, memory consumption, and extended training times present significant challenges, particularly for those with limited computational resources. This thesis explores the application of the QLORA technique to fine-tune the LLAMA2 model with significantly fewer trainable parameters, aiming for efficient training on a single GPU. By quantizing the model to 4-bit precision and introducing low-rank adapters (LoRA) in the linear layers, we achieve performance comparable to the original full-precision model. The model is evaluated using the codeBLEU metric on a dataset of chat completions for coding tasks, demonstrating the efficacy of our approach.

---

## Why Fine-Tuning with QLORA Technique

Fine-tuning large language models is crucial for adapting them to specific tasks and domains. Traditional fine-tuning methods require loading the entire model into memory and updating its parameters, which is computationally expensive and often infeasible on medium-sized GPU resources. The QLORA technique addresses these challenges by quantizing the model's weights to lower precision and introducing low-rank adapters (LoRA) to the linear layers. This approach reduces the model size and the number of trainable parameters, making fine-tuning feasible on limited hardware while maintaining performance

## Quantization Technique and LoRA

### Quantization with Bitsandbytes

**Quantization** reduces the memory and computational demands of language models by representing parameters with fewer bits. QLoRA employs a 4-bit NormalFloat data type, using only 4 bits to encode numbers, creating 16 buckets for model parameters. This method optimizes the storage of normally distributed data, where parameters tend to cluster around zero, maximizing the efficiency of the limited bit representation.

**Blockwise quantization** addresses the issue of extreme values skewing the quantization process. Instead of using a single quantization constant for the entire model, which can be affected by outliers, the model parameters are divided into 64 smaller blocks. Each block has its own quantization constant, reducing the influence of large values and providing more accurate quantization for each block. However, this approach increases memory usage since more constants need to be stored additionally 0.5 bits per parameter.

**Double quantization** further reduces memory usage by quantizing these blockwise constants themselves. After computing the quantization constants for each block (typically stored as FP32 values), these constants are further quantized to 8-bit integers, significantly lowering the memory footprint from 0.5 bits per parameter to approximately 0.127 bits per parameter. This nested quantization approach ensures efficient memory usage while maintaining the model's performance and accuracy.

### Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) is a technique that inserts trainable low-rank matrices into the transformer linear layers of the model. These adapters allow the model to learn task-specific adjustments without modifying the original weights, which remain frozen. The PEFT library facilitates the integration of LoRA into the LLAMA2 model, providing a flexible and efficient mechanism for fine-tuning.


![alt text](images/model_size.png)

## Preprocessing the Datasets

The datasets used for fine-tuning consist of 10000 samples of python code completions related to coding tasks. The preprocessing pipeline involves several key steps:

1. **Prompt Format**:
   To help the model quickly understand and differentiate between system prompt, instruction, inputs, and answers, samples needs to be formatted as shown in the figure below. This structured format enables the model to learn more quickly, significantly reducing the number of samples required for it to understand these distinctions on its own compared to an unformatted approach.

![alt text](images/prompt_format.png)

2. **Distribution of Sequence Length**:finding the distribution of sequence lengths to determine the maximum sequence length.From the Figure max length is 5714, it is not practicle for small resources therefore 90 percentile length is taken, which is close to 512 is taken as max length.

![alt text](images/sequence_len's.png)

3. **Tokenization, Padding and Batching**: It is done automatically by SFTTrainer

## Training with Various Parameter Tweaks and Optimizations

To optimize the fine-tuning process on a single GPU, several techniques are employed:

1. **Gradient Accumulation Steps**: This approach accumulates gradients over multiple steps before updating the model parameters, effectively simulating a larger batch size.
2. **Gradient Checkpointing**: Saves memory by recomputing intermediate activations during the backward pass instead of storing them.
3. **Mixed Precision Training**: Combines 16-bit and 32-bit precision to speed up training and reduce memory usage without compromising model accuracy.
4. **Paged Optimizers**: These optimizers manage memory more efficiently by swapping in and out the required parameter segments.
5. **Gradient Clipping by Normalization**: Normalizing and clipping gradients to prevent exploding gradients, ensuring stable and reliable training
6. **Learning Rate Scheduler**: Dynamically adjusts the learning rate during training to ensure optimal convergence.

## Results of Training

The training process demonstrated that the quantized LLAMA2 model with LoRA layers achieves performance close to that of the full-precision model. Training loss and validation loss were monitored throughout training to ensure the model's effectiveness. The reduced memory footprint and computational requirements facilitated efficient training on a single GPU.

![alt text](images/Trainingresult.png)

## Evaluation Using CodeBLEU

The fine-tuned model's performance was evaluated using the codeBLEU metric, which assesses the quality of generated code completions based on factors like syntax correctness and semantic relevance. The results indicate that the QLORA-enhanced model performs competitively, showcasing its potential for practical applications in code generation tasks.

![alt text](images/codebleu_score.png)

## Conclusion

This research demonstrates that the QLORA technique, combining model quantization with low-rank adaptation, offers a viable solution for fine-tuning large language models on limited hardware resources. By significantly reducing the number of trainable parameters and memory requirements, this approach enables efficient training while maintaining high performance. The successful application of QLORA to the LLAMA2 model for code completion tasks underscores its effectiveness and potential for broader adoption in NLP tasks requiring fine-tuning on specialized datasets.
