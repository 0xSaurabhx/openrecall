from doctr.models import ocr_predictor
import numpy as np
import os
from PIL import Image

ocr = ocr_predictor(
    pretrained=True,
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_mobilenet_v3_large",
)


def extract_text_from_image(image):
    """
    Extract text from an image file path or numpy array.
    
    Args:
        image: Either a file path (string) or numpy array
    
    Returns:
        str: Extracted text
    """
    try:
        print(f"🔍 OCR: Processing image type: {type(image)}")
        
        # Handle both file paths and numpy arrays
        if isinstance(image, str):
            # It's a file path - load it as PIL Image first, then convert to numpy
            print(f"🔍 OCR: Processing file path: {image}")
            if not os.path.exists(image):
                print(f"❌ OCR: File does not exist: {image}")
                return ""
            # Load the image file as PIL Image, then convert to numpy array
            pil_image = Image.open(image)
            image_array = np.array(pil_image)
            # Ensure it's RGB (3 channels)
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Convert to RGB
            elif len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)  # Convert to RGB
            result = ocr([image_array])
        elif isinstance(image, np.ndarray):
            # It's a numpy array - ensure it's in correct format
            print(f"🔍 OCR: Processing numpy array with shape: {image.shape}")
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                result = ocr([image])
            else:
                print(f"Warning: Unexpected image array shape: {image.shape}")
                return ""
        else:
            print(f"Warning: Unexpected image type: {type(image)}")
            return ""
            
        print(f"🔍 OCR: Processing completed, extracting text...")
        text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text += word.value + " "
                    text += "\n"
                text += "\n"
        
        text = text.strip()
        print(f"✅ OCR: Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"❌ Error in OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return ""
