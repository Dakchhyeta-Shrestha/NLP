# A6: Naive RAG vs Contextual Retrieval

## Project Overview

This project implements a domain-specific Question Answering (QA) system
using Retrieval Augmented Generation (RAG). The system is built using
Chapter 11 of the course textbook and answers questions based strictly
on the chapter content.

The main objective of this assignment is to compare two retrieval
strategies:

1.  Naive RAG
2.  Contextual Retrieval

Both approaches retrieve relevant text chunks from the chapter and
generate answers to user questions. Their performance is evaluated using
ROUGE metrics.

------------------------------------------------------------------------

# Project Structure

. ├── main.ipynb ├── README.md ├── app/ │ ├── app.py │ └──
chapter11_chunks.json └── answer/ └── response-st127761-chapter-11.json

### File Description

  ----------------------------------------------------------------------------------------------
  File                                       Description
  ------------------------------------------ ---------------------------------------------------
  main.ipynb                                 Jupyter notebook containing Task 1 and Task 2
                                             implementation

  app/app.py                                 Streamlit chatbot application

  app/chapter11_chunks.json                  Processed chunks used for retrieval

  answer/response-st127761-chapter-11.json   Evaluation results for the 20 QA pairs
  ----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Models Used

## Retriever Model

all-MiniLM-L6-v2 from Sentence Transformers

This model converts text chunks and user queries into vector embeddings
that allow semantic similarity search.

## Generator

The generator produces answers using the retrieved chunks as context.\
The system selects the most relevant chunks and generates a response
based on that information.

------------------------------------------------------------------------

# Task 1: Source Discovery & Data Preparation

Steps performed:

1.  Extracted text from Chapter 11 of the textbook
2.  Cleaned and normalized the text
3.  Split the chapter into smaller text chunks
4.  Created 20 question answer pairs based strictly on the chapter
    content

These chunks were used to build the knowledge base for the RAG system.

------------------------------------------------------------------------

# Task 2: Retrieval Technique Comparison

Two retrieval pipelines were implemented.

## Naive RAG

The Naive RAG pipeline retrieves chunks directly from the vector
database using embedding similarity.

Process:

1.  Convert chunks into embeddings
2.  Store embeddings in ChromaDB
3.  Convert user question into embedding
4.  Retrieve the top k most similar chunks
5.  Generate an answer using the retrieved context

------------------------------------------------------------------------

## Contextual Retrieval

Contextual Retrieval improves the Naive RAG pipeline by adding a
contextual prefix to each chunk before embedding.

Example:

Before contextualization

Revenue grew 40 percent to \$314M with improved margins.

After contextualization

This chunk from the document discusses financial performance and revenue
growth. Revenue grew 40 percent to \$314M with improved margins.

This additional context can help the retriever better understand the
meaning of each chunk.

------------------------------------------------------------------------

# Evaluation Method

Both pipelines were evaluated using 20 question answer pairs.

For each question the system generated:

-   Naive RAG answer
-   Contextual Retrieval answer

The generated answers were compared against the ground truth answers
using ROUGE metrics.

Metrics used:

-   ROUGE-1
-   ROUGE-2
-   ROUGE-L

------------------------------------------------------------------------

# ROUGE Evaluation Results

  Method                 ROUGE-1   ROUGE-2   ROUGE-L
  ---------------------- --------- --------- ---------
  Naive RAG              0.17      0.026     0.13
  Contextual Retrieval   0.16      0.02      0.13


------------------------------------------------------------------------

# Discussion of Results

The evaluation results show that both retrieval methods produced similar
performance across all ROUGE metrics.

Naive RAG slightly outperformed Contextual Retrieval in ROUGE-1 and
ROUGE-2 scores.\
However both approaches produced the same ROUGE-L score.

This indicates that contextual chunk enrichment did not significantly
improve retrieval performance for this dataset.

One possible explanation is that the original chunks already contained
sufficient contextual information, meaning additional context did not
provide a major advantage.

Another factor may be the relatively small dataset used in the
experiment. Contextual retrieval techniques tend to show greater
improvements when applied to larger document collections.

------------------------------------------------------------------------

# Sample Output Screenshots

## Output Example 1

![Screenshot](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-%206/output/output%20-1.png)

Description: Example question answered using the Naive RAG pipeline.

------------------------------------------------------------------------

## Output Example 2

![Screenshot](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-%206/output/output%20-2.png)

Description: Example question answered using the Contextual Retrieval
pipeline.

------------------------------------------------------------------------

## Output Example 3

![Screenshot](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-%206/output/output%20-3.png)

Description: Retrieval results showing the most relevant chunks used for
answer generation.

------------------------------------------------------------------------

## Output Example 4

![Screenshot](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-%206/output/output%20-4.png)

Description: Example of chatbot response generated by the Streamlit
application.

------------------------------------------------------------------------

# Task 3: Chatbot Web Application

A simple chatbot interface was developed using Streamlit.

Features:

-   Users can ask questions related to Chapter 11
-   The system retrieves relevant chunks using Contextual Retrieval
-   The chatbot displays the generated answer and the source chunk used

------------------------------------------------------------------------

# Running the Web Application

cd app streamlit run app.py

------------------------------------------------------------------------

# Conclusion

This project demonstrates how Retrieval Augmented Generation can be used
to build a domain specific QA system.

Two retrieval strategies were compared:

-   Naive RAG
-   Contextual Retrieval

The experiment showed that both methods produced similar ROUGE scores in
this dataset. While contextual retrieval is designed to improve
retrieval quality, its benefits may become more apparent when applied to
larger knowledge bases.

This assignment provided practical experience in building a RAG
pipeline, evaluating retrieval methods, and deploying a simple QA
chatbot.
