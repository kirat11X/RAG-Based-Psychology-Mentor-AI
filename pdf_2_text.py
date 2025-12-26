import os
from pdf2image import convert_from_path
import pytesseract
from reportlab.pdfgen import canvas

def pdf_to_text_pdf(input_path, output_path):
    # Convert PDF pages to images
    pages = convert_from_path(input_path)
    extracted_text = []
    for page in pages:
        text = pytesseract.image_to_string(page)
        extracted_text.append(text)

    # Write OCR text into a new PDF
    c = canvas.Canvas(output_path)
    y = 800
    for page_text in extracted_text:
        for line in page_text.split("\n"):
            c.drawString(50, y, line)
            y -= 15
            if y < 50:  # new page
                c.showPage()
                y = 800
    c.save()

# Process all PDFs in a folder
input_dir = "image_pdf/"
output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"ocr_{filename}")
        pdf_to_text_pdf(in_path, out_path)
        print(f"Processed {filename} → {out_path}")
