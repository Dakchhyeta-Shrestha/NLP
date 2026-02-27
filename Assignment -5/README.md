# Assignment 5  DPO Fine Tuning and Evaluation

## Project overview
This project explores hyperparameter tuning and Direct Preference Optimization DPO to improve a pretrained language model. The workflow included importing a preference dataset, running controlled training experiments, uploading the fine tuned model to Hugging Face, and evaluating the base and DPO models using AlpacaEval with an LLM as a judge. The objective was to analyze how training configurations influence performance and whether DPO improves model quality.

## Dataset
The dataset used for DPO training was `jondurbin/truthy-dpo-v0.1`. This dataset contains prompt, chosen response, and rejected response pairs designed to improve truthfulness and factual consistency. The dataset was imported using the Hugging Face datasets library and prepared for training so it could be used in preference based training.

## Hyperparameter experiments
Before applying DPO, three experiments were conducted to analyze how learning rate, number of epochs, and batch size affect model performance.

### Learning rate experiment
Three learning rates were tested: 1e-3, 1e-4, and 1e-5. In all cases, training loss decreased between epochs, showing that the model learned successfully. Smaller learning rates produced more stable convergence and lower average training loss. The 1e-5 configuration resulted in the lowest loss, suggesting that smaller updates preserve pretrained knowledge better during fine tuning.

Learning rate results screenshot  

![Learning rate results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/Learning_rate_experiment.jpeg)

In this experiment, we evaluated the performance of the model using three different learning rates: **1e-3, 1e-4, and 1e-5**. Across all learning rates, the **training loss** showed steady improvement, with the model learning progressively over the epochs. Specifically, for `lr=1e-3`, the training loss decreased from **1.32 to 1.21**, for `lr=1e-4`, it decreased from **1.08 to 1.01**, and for `lr=1e-5`, it reduced slightly from **0.96 to 0.95**, indicating consistent learning. The training loss consistently improved for all three learning rates, with lr=1e-5 providing the smallest training loss. Moving forward, further experiments with other evaluation metrics, such as **validation loss**, could provide more reliable insights. Additionally, fine-tuning other hyperparameters (such as batch size and number of epochs) or using a learning rate scheduler may help optimize the model’s performance further.

### Epoch experiment
The epoch experiment examined how increasing training iterations affected performance. Training loss improved as epochs increased, indicating progressive learning. However, the improvement became smaller over time, suggesting diminishing returns. Under limited GPU resources, increasing epochs beyond a certain point did not significantly improve validation accuracy.

Epoch results screenshot  

![Epoch results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/epoch_increase_experiment.jpeg)

In this experiment, we tested the performance of the model with different numbers of epochs: **2, 5, and 10**. As the number of epochs increased, the training loss consistently decreased, indicating the model's improvement with additional training.

*   For **Epoch 2**, the training loss started at **1.32** and decreased to **1.21**, with
a validation accuracy of **193.98%**.

*   For **Epoch 5**, the training loss started at **1.14** and dropped to **0.98**, with a validation accuracy of **193.77%.**

*   For **Epoch 10**, the training loss started at **0.99** and dropped to **0.67**, with a validation accuracy of **189.36%.**

While the training loss improved steadily with more epochs, the **validation accuracy** seemed to fluctuate. The validation accuracy was highest after 2 epochs and started decreasing with more epochs, indicating potential **overfitting** after longer training. Based on these results, a **balanced number of epochs** (likely between 2 to 5) appears optimal, as further training beyond that does not yield substantial improvement and may lead to overfitting.

### Batch size experiment
Batch sizes of 4, 8, and 16 were tested. Larger batch sizes reduced training loss more quickly, but validation accuracy was highest with the smallest batch size. This suggests that while larger batches improve optimization efficiency, smaller batches may generalize better. The results highlight a trade off between convergence speed and validation performance.

Batch size results screenshot  

![Batch size results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/Batch_size_experiment.jpeg)

## Hugging Face repository
The final DPO fine tuned model was uploaded to Hugging Face for reproducibility and version control. This ensures that the trained model can be publicly accessed and evaluated.

Paste your Hugging Face model URL below

Hugging Face model URL: [https://huggingface.co/Dakchhyeta/Assignment5-qwen-dpo/tree/main]

In this experiment, we tested the performance of the model using different batch sizes: **4, 8, and 16**. The results show how the batch size influences both **training loss** and **validation accuracy**.
**Batch Size = 4**: The model achieved a training loss of **0.49** after 2 epochs, with a validation accuracy of **189.28%**, showing steady improvement in both metrics across epochs.
**Batch Size = 8**: With this batch size, the training loss decreased to **0.34**, but the validation accuracy was slightly lower at **188.50%**. This suggests that although the training loss decreased, the model did not generalize as well on the validation data.
**Batch Size = 16**: The model showed the lowest training loss of **0.26** with this batch size, but the validation accuracy was **188.21%**, which was lower than the other batch sizes. The larger batch size led to faster training, but it also resulted in a slightly reduced performance on the validation set.
Overall, while the **training loss** improved with increasing batch size, the **validation accuracy** was highest with a **batch size of 4**, suggesting a trade-off between faster training (with larger batch sizes) and better generalization (with smaller batch sizes).

## DPO training
The base model used was `Qwen/Qwen2.5-0.5B-Instruct`. This model was fine tuned using DPO with the truthy dpo dataset. Due to GPU memory constraints and limited training time, the model was trained for a small number of epochs with reduced batch sizes. Although this limited extensive hyperparameter tuning, a stable DPO model was produced and uploaded.

## Evaluation using AlpacaEval
Evaluation was conducted using the `tatsu-lab/alpaca_eval` dataset, specifically the `helpful_base` subset. Fifteen prompts were randomly sampled. Responses were generated using both the base model Model A and the DPO model Model B. Gemini was used as the judge through the web interface, and it selected the better response for each prompt based on helpfulness and accuracy.

All 15 prompts were judged in favor of the base model.

Model B wins: 0  
Ties: 0  
Total evaluations: 15  
Win rate: 0 percent

## Discussion
The DPO model did not outperform the base model on the AlpacaEval helpful_base benchmark. One possible explanation is dataset mismatch: the truthy dpo dataset focuses on factual correctness, while AlpacaEval measures general helpfulness and instruction quality. Additionally, limited GPU resources restricted training duration and hyperparameter tuning, which may have affected convergence. These results show that preference optimization must be aligned with the evaluation objective to produce measurable improvements.

## How to reproduce
Run the notebook in order. 
First load and prepare the truthy dpo dataset. 
Next run the hyperparameter experiments for learning rate, epochs, and batch size. 
Then train the DPO model and save it. 
Finally load AlpacaEval helpful_base, sample 15 prompts, generate responses from both models, and judge them using Gemini.
