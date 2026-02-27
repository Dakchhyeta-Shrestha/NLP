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
Paste your image link below

![Learning rate results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/Learning_rate_experiment.jpeg)

### Epoch experiment
The epoch experiment examined how increasing training iterations affected performance. Training loss improved as epochs increased, indicating progressive learning. However, the improvement became smaller over time, suggesting diminishing returns. Under limited GPU resources, increasing epochs beyond a certain point did not significantly improve validation accuracy.

Epoch results screenshot  
Paste your image link below

![Epoch results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/epoch_increase_experiment.jpeg)

### Batch size experiment
Batch sizes of 4, 8, and 16 were tested. Larger batch sizes reduced training loss more quickly, but validation accuracy was highest with the smallest batch size. This suggests that while larger batches improve optimization efficiency, smaller batches may generalize better. The results highlight a trade off between convergence speed and validation performance.

Batch size results screenshot  
Paste your image link below

![Batch size results](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-5/Screenshots/Batch_size_experiment.jpeg)

## Hugging Face repository
The final DPO fine tuned model was uploaded to Hugging Face for reproducibility and version control. This ensures that the trained model can be publicly accessed and evaluated.

Paste your Hugging Face model URL below

Hugging Face model URL: [https://huggingface.co/Dakchhyeta/Assignment5-qwen-dpo/tree/main]

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
