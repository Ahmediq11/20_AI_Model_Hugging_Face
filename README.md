# 20_AI_Model_Hugging_Face

# Hugging Face Transformers: A Practical Tour 🚀

Welcome to this collection of Jupyter notebooks designed to be a practical guide to the amazing capabilities of the Hugging Face `transformers` library\! This project demonstrates how to use pre-trained models for a wide variety of tasks across different domains, including Natural Language Processing (NLP), Computer Vision, and Audio.

Each notebook is a self-contained example that is easy to follow and includes detailed explanations for every step.

## 📋 Table of Contents

1.  [Getting Started](https://www.google.com/search?q=%23-getting-started)
2.  [Notebook Descriptions](https://www.google.com/search?q=%23-notebook-descriptions)
      - [1. Text Classification](https://www.google.com/search?q=%231-text-classification-emotion-detection-)
      - [2. Named Entity Recognition (NER)](https://www.google.com/search?q=%232-named-entity-recognition-ner-for-keyphrase-extraction-)
      - [3. Question Answering](https://www.google.com/search?q=%233-extractive-question-answering-)
      - [4. Text Summarization](https://www.google.com/search?q=%234-abstractive-text-summarization-)
      - [5. Machine Translation](https://www.google.com/search?q=%235-machine-translation-)
      - [6. Image Classification](https://www.google.com/search?q=%236-image-classification-)
      - [7. Image Segmentation](https://www.google.com/search?q=%237-image-segmentation-)
      - [8. Text-to-Speech (TTS)](https://www.google.com/search?q=%238-text-to-speech-tts-)
3.  [How to Run the Notebooks](https://www.google.com/search?q=%23-how-to-run-the-notebooks)
4.  [Contribution](https://www.google.com/search?q=%23-contribution)

-----

## 🚀 Getting Started

This project uses Python and several libraries from the Hugging Face ecosystem. To get started, you'll need to have Python installed. Each notebook includes the necessary `pip install` commands to set up the environment for that specific task.

The core dependencies across all notebooks are:

  - `transformers`: The main library from Hugging Face.
  - `torch`: The backend deep learning framework.
  - `datasets`: For loading datasets (especially for audio and voice embeddings).

-----

## 📖 Notebook Descriptions

Here's a breakdown of what you can learn from each notebook.

### 1\. Text Classification (Emotion Detection) 🎭

  - **File**: `1-text-classification.ipynb`
  - **Description**: This notebook shows how to classify a piece of text to detect the underlying emotion.
  - **Model Used**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - **Key Concepts**: `pipeline` for `text-classification`, using a model fine-tuned for emotion detection on social media text.

\<hr\>

### 2\. Named Entity Recognition (NER) for Keyphrase Extraction 🔑

  - **File**: `2-ner.ipynb`
  - **Description**: This notebook demonstrates how to use a Named Entity Recognition (NER) model to perform keyphrase extraction, which is useful for identifying the most important terms in a document.
  - **Model Used**: A selection including `dslim/bert-base-NER`.
  - **Key Concepts**: `pipeline` for `ner`, `aggregation_strategy` to group word pieces into coherent entities.

\<hr\>

### 3\. Extractive Question Answering ❓

  - **File**: `3-question-answering.ipynb`
  - **Description**: Learn how to ask a model a question about a given piece of text and have it extract the answer directly from the context.
  - **Model Used**: `deepset/roberta-base-squad2`
  - **Key Concepts**: `pipeline` for `question-answering`, providing both a `context` and a `question` to the model.

\<hr\>

### 4\. Abstractive Text Summarization ✍️

  - **File**: `4-summarization.ipynb`
  - **Description**: This notebook shows how to generate a concise summary of a longer document. The model generates new sentences to capture the core meaning, rather than just copying existing ones.
  - **Model Used**: A selection including `facebook/bart-large-cnn`.
  - **Key Concepts**: `pipeline` for `summarization`, understanding the difference between abstractive and extractive summarization.

\<hr\>

### 5\. Machine Translation 🌍

  - **File**: `5-translation.ipynb`
  - **Description**: A practical example of translating text from one language (English) to another (Arabic).
  - **Model Used**: `Helsinki-NLP/opus-mt-en-ar`
  - **Key Concepts**: `pipeline` for translation tasks (e.g., `translation_en_to_ar`), using specialized models from the `Helsinki-NLP` group.

\<hr\>

### 6\. Image Classification 👁️

  - **File**: `6-Image-Classification.ipynb`
  - **Description**: This notebook steps into the world of computer vision, showing how to classify the content of an image using a Vision Transformer (ViT) model.
  - **Model Used**: `google/vit-base-patch16-224` (default model)
  - **Key Concepts**: `pipeline` for `image-classification`, using the `Pillow` library to handle images.

\<hr\>

### 7\. Image Segmentation 🎨

  - **File**: `7-Image-Segmentation.ipynb`
  - **Description**: Go beyond classification to identify *where* objects are in an image on a pixel-by-pixel basis. This example segments different articles of clothing on a person.
  - **Model Used**: `mattmdjaga/segformer_b2_clothes`
  - **Key Concepts**: `pipeline` for `image-segmentation`, understanding segmentation masks as output.

\<hr\>

### 8\. Text-to-Speech (TTS) 🗣️

  - **File**: `8-Text_to_speech.ipynb`
  - **Description**: This notebook demonstrates how to convert text into realistic-sounding speech. It also shows how to control the voice of the generated audio by using speaker embeddings.
  - **Model Used**: `microsoft/speecht5_tts`
  - **Key Concepts**: `pipeline` for `text-to-speech`, loading speaker embeddings from the `datasets` library to control voice characteristics, and saving audio to a `.wav` file.

-----

## ⚙️ How to Run the Notebooks

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Open the notebooks:**
    You can use Jupyter Notebook, JupyterLab, or Google Colab to open and run the `.ipynb` files.

3.  **Install dependencies:**
    Each notebook begins with a code cell (often commented out) that contains the `pip install` commands needed for that specific script. Simply run this cell to install the required libraries.

4.  **Run the cells:**
    Execute the cells in order from top to bottom. The markdown cells provide detailed explanations of what the code is doing.

> **Note on Cache Directory**: The notebooks include an optional section to set a custom cache directory for Hugging Face models. This is highly recommended to avoid filling up your user home folder with large model files. Make sure to change the path to a directory that exists on your machine.

-----

## 🙌 Contribution

Contributions are welcome\! If you have an idea for a new example notebook or an improvement to an existing one, feel free to open an issue or submit a pull request.
