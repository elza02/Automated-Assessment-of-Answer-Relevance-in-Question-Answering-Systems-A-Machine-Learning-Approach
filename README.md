# Automated Assessment of Answer Relevance in QA Systems

This repository contains the implementation of a neural model designed to assess the relevance of answers in French-language question-answering systems. The model was developed as part of a Kaggle-style hackathon and is documented in detail in the accompanying paper.

## Overview

The project focuses on scoring the relevance of answers to French-language queries using a hierarchical architecture based on the pre-trained CamemBERT model. The solution leverages a combination of dimensional reduction, dropout regularization, and layer normalization to produce relevance scores on a scale from 0 to 1.

## Key Features

- **Base Model**: CamemBERT (`camembert-base`) for contextual French language embeddings.
- **Data Source**: French Question Answering Dataset (FQuAD).
- **Balanced Dataset**: Positive and negative pairs were generated for effective training.
- **Loss Function**: Binary Cross-Entropy with Logits Loss.
- **Optimization**: AdamW optimizer with weight decay correction.
- **Training**: Mixed precision training for memory efficiency and faster computation.

## Model Architecture

The architecture includes:

1. **Base Language Model**: CamemBERT for extracting embeddings.
2. **Feature Processing Layers**:
   - Dense layer to reduce dimensions.
   - Dropout layer for regularization.
   - Layer normalization for stabilized training.
3. **Output Layer**: Single-unit layer producing raw logits, converted into relevance scores via the sigmoid function.

## Inference Pipeline

1. Tokenize input question-article pairs.
2. Process the pairs through the trained model.
3. Apply sigmoid activation to output logits.
4. Return a relevance score between 0 and 1.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone [repository_url]
cd [repository_name]
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run the model on custom question-answer pairs:

1. Prepare the input data in the required format.
2. Use the provided script to load the model and make predictions:

```python
python predict.py --input_file your_input_file.json --output_file predictions.json
```

## Dataset

The training data is based on FQuAD, with positive pairs (original question-article pairs) labeled as relevant and negative pairs (randomly sampled unrelated articles) labeled as irrelevant.

## Results

The model produces continuous relevance scores that effectively capture the semantic relationships between questions and potential answers while maintaining computational efficiency.

## Contributions

Developed by: **Zakaria El Alaoui**

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or feedback, feel free to contact me at zakaria.elalaoui@edu.uiz.ac.ma.
