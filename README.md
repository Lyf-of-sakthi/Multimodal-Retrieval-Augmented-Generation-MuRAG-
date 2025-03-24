# Multimodal RAG Assistant

## Overview
The **Multimodal Retrieval-Augmented Generation (RAG) Assistant** is a system that extracts and retrieves relevant text and image information from PDFs and generates responses based on user queries. It combines **DeepSeek-R1-Distill-Qwen-1.5B** for text processing and **BLIP** for image captioning.

## Features
- **PDF Processing**: Extracts text and images from PDFs using PyMuPDF (fitz).
- **Image Captioning**: Uses BLIP to generate captions for extracted images.
- **Vector Search with FAISS**: Stores and retrieves relevant text and image captions efficiently.
- **Multimodal Querying**: Supports both text-based and image-related queries.
- **LLM-powered Response**: Uses DeepSeek-R1-Distill-Qwen-1.5B for text generation.
- **Flet UI**: Provides an interactive web-based interface.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- CUDA (for GPU acceleration, optional)

### Dependencies
Install the required packages using:
```bash
pip install -r requirements.txt
```

### GPU Support
If using a GPU, install the necessary PyTorch version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Run the Application
```bash
python app.py
```

### 2. Upload PDF Files
- Use the UI to upload a PDF.
- The system extracts text and images automatically.

### 3. Query the Assistant
- Enter a text query to retrieve relevant information.
- The system combines retrieved text and image captions to generate responses.

## Architecture
1. **PDF Parsing**: Extracts text and images from PDFs using PyMuPDF.
2. **Image Captioning**: BLIP generates captions for extracted images.
3. **Vector Search**: FAISS stores text and image embeddings for retrieval.
4. **LLM Response Generation**: DeepSeek-R1-Distill-Qwen-1.5B model processes queries and generates responses.
5. **UI Interface**: Flet provides an interactive front-end for users.

## Configuration
Modify `config.py` to customize settings:
```python
PDF_DIR = "data/pdf_files"
VECTOR_DB_PATH = "data/faiss_index"
LLM_MODEL = "DeepSeek-R1-Distill-Qwen-1.5B"
IMAGE_CAPTION_MODEL = "BLIP"
```

## Roadmap
- [ ] Implement multilingual support
- [ ] Improve response ranking with Rerankers
- [ ] Optimize UI with enhanced filtering options

## License
This project is licensed under the MIT License.

## Acknowledgments
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai)
- [BLIP Model](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Flet](https://flet.dev/)

