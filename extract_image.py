import fitz
from PIL import Image
import pytesseract
import io

def extract_image_from_pdfs(file_paths):
  for file_path in file_paths:
    pdf = fitz.open(file_path)

    for page_num in range(pdf.page_count):
      page = pdf.load_page(page_num)
      # Extract text
      text = page.get_text("text")
      image_metadata_dict = {}
      # Extract images
      images = page.get_images()
      print(page_num, images)
      for img_index, img in enumerate(images):
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))
        
        # OCR
        # image_text = pytesseract.image_to_string(image)
        
        #Save or process the image and its text
        img_path = "doc/image/page{page_num+1}_img{img_index+1}.{image_ext}"
        image.save(img_path)
        # print(f"Text from image {img_index+1}: {image_text}")
        image_metadata_dict["img_path"] = img_path
        # TODO
        # image_metadata_dict["img_discription"] = image_discription
  return image_metadata_dict
        
extract_image_from_pdfs(["data/Deep_Residual_Learning.pdf"])