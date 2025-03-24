import flet as ft
from flet import Page, Text, TextField, ElevatedButton, Column, Row, Container, Image as FletImage, padding, colors
import fitz 
from PIL import Image
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys
import asyncio
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tempfile
import os
import base64
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS as LC_FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
text_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
text_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
image_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
image_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return full_text, images

def generate_image_captions(images):
    captions = []
    for image_bytes in images:
        img = Image.open(io.BytesIO(image_bytes))
        inputs = image_processor(img, return_tensors="pt")
        outputs = image_model.generate(**inputs)
        caption = image_processor.decode(outputs[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def create_faiss_index(doc_text, captions):
    all_texts = [doc_text] + captions
    embeddings = embedding_model.encode(all_texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, all_texts
import re

def refine_answer(answer, query):
    answer = re.sub(
        r"Use the following pieces of context.*?(?=\n|$)",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    if "heart" in query.lower():
        lines = []
        for line in answer.split("\n"):
            if "brain" not in line.lower():
                lines.append(line)
        answer = "\n".join(lines).strip()

    if answer and answer[-1] not in ".!?":
        if "." in answer:
            answer = answer.rsplit(".", 1)[0] + "."
        else:
            answer += "."
    answer = re.sub(r"\n\s*\n+", "\n", answer).strip()

    return answer

from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def generate_response(query, context):
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    completion = client.chat.completions.create(
         model="deepseek-ai/deepseek-r1",
         messages=[{"role": "user", "content": prompt}],
         temperature=0.3,
         top_p=0.5,
         max_tokens=150,
         stream=False
    )

    response = completion.choices[0].message.content.strip()
    
    response = re.sub(r"</think>.*$", "", response, flags=re.IGNORECASE | re.DOTALL).strip()
    
    response = re.sub(r"^.*?(?=[A-Z])", "", response, flags=re.DOTALL).strip()
    
    if response and response[-1] not in ".!?":
        if "." in response:
            response = response.rsplit(".", 1)[0] + "."
        else:
            response += "."
    return response



def bytes_to_temp_image(img_bytes):
    temp_path = os.path.join(tempfile.gettempdir(), "result_image.png")
    with open(temp_path, "wb") as f:
        f.write(img_bytes)
    return temp_path

def split_text(text, max_words=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def process_text_query(query, doc_text):
    chunks = split_text(doc_text, max_words=100)
    docs = [Document(page_content=chunk) for chunk in chunks]
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = LC_FAISS.from_documents(docs, hf_embeddings)

    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    answer = generate_response(query, context)
    return answer


def process_image_query(query, captions, orig_images):
    docs = []
    for i, cap in enumerate(captions):
        docs.append(Document(page_content=cap, metadata={"index": i}))
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = LC_FAISS.from_documents(docs, hf_embeddings)
    results = vectorstore.similarity_search(query, k=1)
    if results:
        best_doc = results[0]
        best_index = best_doc.metadata.get("index", None)
        if best_index is not None and best_index < len(orig_images):
            return best_doc.page_content, orig_images[best_index]
    return "No relevant image found.", None

#flet
import flet as ft
import base64

def main(page: ft.Page):
    page.title = "Multimodal RAG System"
    page.bgcolor = ft.colors.GREY_50
    page.padding = 30
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.colors.INDIGO,
            secondary=ft.colors.PINK,
            surface=ft.colors.WHITE,
        ),
        visual_density=ft.VisualDensity.COMPACT,
    )


    page.session.set("doc_text", "")
    page.session.set("orig_images", [])
    page.session.set("captions", [])


    header = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.icons.ASSISTANT_SHARP, size=40, color=ft.colors.INDIGO),
                ft.Text("Multimodal RAG Assistant", size=32, weight="bold", color=ft.colors.INDIGO_900),
            ],
            alignment="center",
            spacing=15
        ),
        margin=ft.margin.only(bottom=30),
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=15, color=ft.colors.BLUE_100)
    )


    upload_status = ft.Row(
        controls=[
            ft.Icon(ft.icons.CHECK_CIRCLE, color=ft.colors.GREEN, visible=False),
            ft.Text("", size=14, color=ft.colors.GREY_600)
        ],
        spacing=5,
        visible=False
    )

    file_display = ft.Text("No PDF selected", italic=True, color=ft.colors.GREY_600)
    
    upload_btn = ft.FilledTonalButton(
        content=ft.Row(
            [ft.Icon(ft.icons.CLOUD_UPLOAD), ft.Text("Choose PDF")],
            alignment="center",
            spacing=5
        ),
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=20,
            overlay_color=ft.colors.TRANSPARENT
        ),
        on_click=lambda e: file_picker.pick_files(),
    )

 
    text_query = ft.TextField(
        label="Text Query",
        prefix_icon=ft.icons.TEXT_SNIPPET,
        filled=True,
        color=ft.colors.BLACK,
        border_radius=15,
        border_color=ft.colors.TRANSPARENT,
        focus_color=ft.colors.INDIGO_100,
        bgcolor=ft.colors.GREY_100,
        width=400
    )

    image_query = ft.TextField(
        label="Image Query", 
        prefix_icon=ft.icons.IMAGE_SEARCH,
        filled=True,
        color=ft.colors.BLACK,
        border_radius=15,
        border_color=ft.colors.TRANSPARENT,
        focus_color=ft.colors.INDIGO_100,
        bgcolor=ft.colors.GREY_100,
        width=400
    )


    text_response = ft.Column(
        [
            ft.Row([ft.Icon(ft.icons.ARTICLE, color=ft.colors.INDIGO), ft.Text("Text Response", weight="bold")], spacing=10),
            ft.Container(
                ft.Text("", selectable=True, color=ft.colors.BLACK),
                padding=15,
                border=ft.border.all(1, ft.colors.INDIGO_100),
                border_radius=15,
                bgcolor=ft.colors.WHITE,
                width=400,
                height=300
            )
        ],
        spacing=10
    )

    image_response = ft.Column(
        [
            ft.Row([ft.Icon(ft.icons.IMAGE, color=ft.colors.INDIGO), ft.Text("Image Response", weight="bold")], spacing=10),
            ft.Container(
                content=ft.Column([], alignment="center"),
                padding=15,
                border=ft.border.all(1, ft.colors.INDIGO_100),
                border_radius=15,
                bgcolor=ft.colors.WHITE,
                width=400,
                height=300,
                alignment=ft.alignment.center
            )
        ],
        spacing=10
    )


    submit_btn = ft.ElevatedButton(
        content=ft.Row(
            [ft.Icon(ft.icons.SEND), ft.Text("Process Queries")],
            spacing=10,
            alignment="center"
        ),
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=20,
            bgcolor=ft.colors.INDIGO,
            overlay_color=ft.colors.INDIGO_100,
        ),
        color=ft.colors.WHITE,
    )

    loading_indicator = ft.Row(
        [
            ft.ProgressRing(width=20, height=20, stroke_width=2, color=ft.colors.INDIGO),
            ft.Text("Processing...", color=ft.colors.GREY_600)
        ],
        alignment="center",
        visible=False
    )


    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    def on_file_pick(e):
        if e.files:
            path = e.files[0].path
            file_display.value = path.split("/")[-1]
            upload_status.controls[1].value = "PDF uploaded successfully"
            upload_status.controls[0].visible = True
            upload_status.visible = True
            
            doc_text, orig_images = extract_text_and_images(path)
            caps = generate_image_captions(orig_images)
            
            page.session.set("doc_text", doc_text)
            page.session.set("orig_images", orig_images)
            page.session.set("captions", caps)
            
            page.update()

    file_picker.on_result = on_file_pick

    
    def handle_submit(e):
        loading_indicator.visible = True
        page.update()
        
        try:

            text_answer = process_text_query(text_query.value, page.session.get("doc_text"))
            text_response.controls[1].content.value = text_answer
            

            img_caption, img_bytes = process_image_query(
                image_query.value,
                page.session.get("captions"),
                page.session.get("orig_images")
            )
            
            image_container = image_response.controls[1].content
            image_container.controls = []
            
            if img_bytes:
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                image_container.controls.extend([
                    ft.Image(src_base64=b64_img, width=380, height=220, fit="contain"),
                    ft.Container(
                        ft.Text(img_caption, size=14, color=ft.colors.BLACK, selectable=True),
                        padding=10,
                        bgcolor=ft.colors.GREY_100,
                        border_radius=10,
                        width=380
                    )
                ])
            else:
                image_container.controls.append(
                    ft.Text("No relevant image found", italic=True)
                )
                
            page.update()
            
        except Exception as ex:
            text_response.controls[1].content.value = f"Error: {str(ex)}"
        
        loading_indicator.visible = False
        page.update()

    submit_btn.on_click = handle_submit

    # Layout
    layout = ft.Column(
        [
            header,
            ft.Row([upload_btn, file_display], spacing=20, alignment="center"),
            upload_status,
            ft.Row([text_query, image_query], spacing=30, alignment="center"),
            ft.Column([submit_btn, loading_indicator], spacing=15, horizontal_alignment="center"),
            ft.Divider(height=30, color=ft.colors.TRANSPARENT),
            ft.Row([text_response, image_response], spacing=30, alignment="center")
        ],
        spacing=25,
        horizontal_alignment="center",
        scroll=ft.ScrollMode.AUTO
    )

    page.add(layout)

ft.app(target=main)