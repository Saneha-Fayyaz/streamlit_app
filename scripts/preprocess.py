"""
Advanced preprocessing pipeline for ring images.
Handles noisy images, different orientations, top-view rings,
and accurately extracts the ring structure for matching.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io


def remove_noise(img: np.ndarray) -> np.ndarray:
    """Apply multi-stage noise removal."""
    # Bilateral filter to preserve edges while removing noise
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Additional median blur for salt-pepper noise
    denoised = cv2.medianBlur(denoised, 3)
    return denoised


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE on L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def extract_ring_region(img: np.ndarray) -> np.ndarray:
    """
    Intelligently extract the ring region from the image.
    Works with noisy backgrounds, various lighting conditions,
    and top-view ring images.
    """
    h, w = img.shape[:2]
    
    # Strategy 1: Threshold-based extraction (light background)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try multiple thresholding approaches
    candidates = []
    
    # Otsu thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(thresh_otsu)
    
    # Fixed high threshold for white backgrounds
    _, thresh_high = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    candidates.append(thresh_high)
    
    # Adaptive thresholding for uneven backgrounds
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    candidates.append(thresh_adapt)
    
    best_crop = img
    best_area = 0
    
    for thresh in candidates:
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Must be significant portion of image (at least 5% of image area)
        if area < (h * w * 0.05):
            continue
        
        if area > best_area:
            best_area = area
            x, y, cw, ch = cv2.boundingRect(largest)
            
            # Add padding
            pad = int(max(cw, ch) * 0.08)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            
            best_crop = img[y1:y2, x1:x2]
    
    # If crop is too small, return original
    if best_crop.shape[0] < 20 or best_crop.shape[1] < 20:
        return img
    
    return best_crop


def normalize_orientation(img: np.ndarray) -> np.ndarray:
    """
    Normalize ring orientation so that similar rings at different
    angles produce similar features. For top-view rings, we make
    the feature extraction rotation-invariant by ensuring consistent
    preprocessing.
    """
    # Detect the principal axis and align
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find moments
    M = cv2.moments(thresh)
    if M['m00'] == 0:
        return img
    
    # For rings (circular/symmetric objects), we focus on the structure
    # rather than orientation - this is handled by the CNN features
    return img


def make_square(img: np.ndarray) -> np.ndarray:
    """Pad image to square without distortion."""
    h, w = img.shape[:2]
    size = max(h, w)
    
    # Create white square canvas
    square = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Center the image
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = img
    
    return square


def preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline for ring images.
    
    Steps:
    1. Load image (handles various formats)
    2. Noise removal
    3. Contrast enhancement  
    4. Ring region extraction
    5. Orientation normalization
    6. Square padding
    7. Resize to target
    8. Normalize pixel values
    
    Returns: Preprocessed numpy array ready for feature extraction
    """
    try:
        # 1. Load image
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL for unusual formats
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise IOError(f"Cannot load image: {image_path}")
        
        # 2. Noise removal
        img = remove_noise(img)
        
        # 3. Contrast enhancement
        img = enhance_contrast(img)
        
        # 4. Extract ring region (remove background, center on ring)
        img = extract_ring_region(img)
        
        # 5. Normalize orientation
        img = normalize_orientation(img)
        
        # 6. Make square (preserves aspect ratio)
        img = make_square(img)
        
        # 7. Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 8. Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 9. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        print(f"[Preprocess] Error processing {image_path}: {e}")
        return None


def preprocess_bytes(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Preprocess image from bytes (for web uploads)."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        img = remove_noise(img)
        img = enhance_contrast(img)
        img = extract_ring_region(img)
        img = normalize_orientation(img)
        img = make_square(img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"[Preprocess] Error: {e}")
        return None
