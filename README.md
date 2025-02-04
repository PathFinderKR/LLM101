# LLM101

**Training Language Models from scratch**


## Table of Contents

- [Overview](#Overview)
- [Quick Start](#🚀Quick-Start)
- [Repository Structure](#📂-Repository-Structure)
- [Installation](#Installation)
- [Usage](#Usage)
  - [Pre-Training](#Pre-Training)
  - [Fine-Tuning](#Fine-Tuning)
  - [Generation](#Generation)
  - [Evaluation](#Evaluation)
- [Models](#Models)
- [License](#⚖️License)
- [Citation](#📜Citation)
- [Acknowledgements](#🙌Acknowledgements)


## Overview
**LLM101** is an educational repository for training simple language models from scratch. It serves as a practical guide for understanding and implementing foundational NLP models such as **Bigram, GPT, and Megabyte**. The repository provides scripts for tokenization, training, fine-tuning, inference, and evaluation, making it a valuable resource for those interested in language modeling and deep learning.


## 🚀 Quick Start
To get started with **LLM101**, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/PathFinderKR/LLM101.git
```
2. Navigate to the project directory:
```bash
cd LLM101
```
3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
4. Install the dependencies:
```bash
pip install
```
5. Build vocabulary:
```bash
python src/tokenizer.py
```
6. Train a model:
```bash
python src/train.py
```
7. Generate text:
```bash
python src/generate.py
```


## 📂 Repository Structure
```plaintext
LLM101/
├── assets/
├── configs/
│   ├── train.yaml
│   ├── generate.yaml
├── data/
│   ├── raw/
│   ├── processed/
├── models/
│   ├── bigram/
│   │   ├── MODEL_CARD.md
│   │   ├── bigram.py
│   │   ├── bigram.yaml
│   │   ├── checkpoints/
│   ├── gpt/
│   │   ├── MODEL_CARD.md
│   │   ├── gpt.py
│   │   ├── gpt.yaml
│   │   ├── checkpoints/
│   ├── megabyte/
│   │   ├── MODEL_CARD.md
│   │   ├── megabyte.py
│   │   ├── megabyte.yaml
│   │   ├── checkpoints/
├── notebooks/
│   ├── Lecture_01.ipynb
│   ├── Lecture_02.ipynb
├── src/
│   ├── tokenizer.py
│   ├── train.py
│   ├── fine_tune.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── utils.py
├── .env
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── char_vocab.json
```


## Installation
To install the dependencies, follow these steps:
```bash
pip install -r requirements.txt
```
# Clone the repository
git clone https://github.com/PathFinderKR/LLM101.git
cd LLM101

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install 
```


## Usage
### Tokenization
```bash
python src/tokenizer.py --type char --text_file /data/raw/shakespeare.txt
```

### Pre-Training
```bash
python src/train.py --train_config configs/train.yaml --model_config models/gpt/config.yaml --vocab_file char_vocab.json
```

### Fine-tuning
```bash
python src/fine_tune.py
```

### Generation
```bash
python src/inference.py --generate_config configs/generate.yaml --model_config models/gpt/config.yaml --vocab_file char_vocab.json
```


## Models


## Todos
- [ ] Add Bigram model
- [ ] Add Megabyte model
- [ ] Add GPT model
- [ ] Add evaluation metrics


## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📜 Citation
If you find this repository useful in your research, please cite the following paper:
```
@article{LLM101,
  title={LLM101: Training Language Models from Scratch},
  author={PathFinderKR},
  year={2025}
}
```


## 🙌 Acknowledgements
Special thanks to