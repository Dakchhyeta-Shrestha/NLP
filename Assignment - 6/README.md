# A6: Naive RAG vs Contextual Retrieval

## Overview
This project implements a domain specific question answering system based on Chapter 11 of the assigned textbook, "Information Retrieval and Retrieval Augmented Generation".

The project compares two retrieval methods:
1. Naive RAG
2. Contextual Retrieval

The performance of both methods is evaluated using ROUGE-1, ROUGE-2, and ROUGE-L.

## Files
- `main.ipynb` : contains Task 1 and Task 2 implementation
- `app/app.py` : Streamlit web application for Task 3
- `app/chapter11_chunks.json` : processed chunks used by the chatbot
- `answer/response-st127761-chapter-11.json` : evaluation results for 20 QA pairs

## Models Used
### Retriever model
`all-MiniLM-L6-v2` from Sentence Transformers

### Generator
A context based answer generation function using retrieved chunks

## Task 1
- Extracted and cleaned text from Chapter 11
- Split the chapter into chunks
- Created 20 QA pairs based strictly on the chapter

## Task 2
- Implemented Naive RAG
- Implemented Contextual Retrieval
- Evaluated both methods using ROUGE scores

## Task 3
- Developed a simple Streamlit chatbot
- The chatbot uses Contextual Retrieval in the backend
- The chatbot displays the generated answer and the source chunk used

## How to Run the App
```bash
cd app
streamlit run app.py