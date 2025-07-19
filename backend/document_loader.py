from PyPDF2 import PdfReader
from docx import Document as DocxDocument

def load_documents(files):
    texts = []
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                texts.append(page.extract_text())
        elif file.name.endswith(".docx"):
            doc = DocxDocument(file)
            for para in doc.paragraphs:
                texts.append(para.text)
        elif file.name.endswith(".txt"):
            texts.append(file.read().decode("utf-8"))
    return texts