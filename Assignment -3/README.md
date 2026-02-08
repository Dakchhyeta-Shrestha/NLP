# Assignment 3: Make Your Own Machine Translation Language

This project implements an **English to Nepali Neural Machine Translation (NMT)** system using a **Seq2Seq Transformer architecture** with three different attention mechanisms:
- General Attention
- Multiplicative Attention
- Additive Attention

The models were trained and evaluated using the **OPUS-100 (English–Nepali)** dataset. A simple web interface was built using **Dash** to demonstrate the translation capability of the best performing model.

---

## Observations during Training and Validation

| Attentions | Training Loss | Training PPL | Validation Loss | Validation PPL |
|-----------|---------------|--------------|-----------------|----------------|
| General Attention | 2.141 | 8.51 | 3.158 | 23.53 |
| Additive Attention | 2.112 | 8.27 | 3.149 | 23.32 |

Additive Attention achieved slightly lower **training loss and perplexity**, indicating better learning during optimisation. It also performed marginally better on the validation set, suggesting improved generalisation to unseen data.

Although the difference between the two models is small, Additive Attention shows **consistent improvement across both training and validation metrics**, making it the preferred choice for deployment.
Due to time constraint and overworked mps device, the training data size is quite small hence, the output might not be what we expected.

---

## Observations during Testing

Both models were evaluated on unseen test data. Overall translation quality remained limited, which may be attributed to:
- A relatively small number of training epochs
- Model overfitting due to limited dataset exposure
- Subword tokenisation producing fragmented outputs for single-step decoding

Despite this, Additive Attention still produced the **lowest loss and perplexity**, reinforcing its selection for the final application.

---

## Performance Analysis

- **Model Size:** All models have similar storage requirements since they share the same base architecture.
- **Inference Time:** Inference time for all attention variants was low and suitable for real-time translation.
- **Attention Behaviour:** Additive Attention demonstrated clearer alignment patterns in attention heatmaps compared to General Attention.

---

## Attention Heatmap Visualisation

To better understand how the model aligns source and target tokens, attention heatmaps were generated.  
For visual clarity, an **8 × 8 attention heatmap** was produced by decoding multiple target steps during inference.

### Additive Attention graph
![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/additive.png)

### Additive Attention Heatmap
![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/Additive_HM.png)

### General Attention graph
![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/general.png)

### General Attention Heatmap
![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/General_HM.png)

These heatmaps illustrate how the decoder attends to different source tokens while generating each target token.

### Website Output

1. ![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/Output-1.png)
2. ![](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-3/code/assests/Output-2.png)

---

## Web Application Demo

The translation system is integrated into a web interface built using **Dash**.

### Features
- Text input field for English sentences
- Translate button with basic validation
- Display of translated Nepali output
- Visual attention heatmap for interpretability

Screenshots of the working application are provided in the `README.md` file and inside the `static` folder.

---

## User Interaction Flow

1. User enters an English sentence
2. User clicks the **Translate** button
3. The translated Nepali output is displayed
4. The corresponding attention heatmap is shown below the result

---

## Conclusion

Among the evaluated attention mechanisms, **Additive Attention** consistently achieved the best performance in terms of loss and perplexity. Although overall translation quality can be improved with additional training and tuning, the results demonstrate the effectiveness of attention-based Transformer models for low-resource language translation tasks.

---

## Technologies Used

- Python
- PyTorch
- TorchText
- HuggingFace Datasets (OPUS-100)
- Dash & Dash Bootstrap Components
- Matplotlib

