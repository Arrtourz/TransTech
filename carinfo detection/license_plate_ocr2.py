import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(denoised)
    
    # Try different threshold methods
    _, thresh1 = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return [thresh1, thresh2, contrast]  # Return multiple processed versions

def extract_alphanumeric(text):
    """
    Extract alphanumeric characters from text and validate against common patterns
    """
    # Remove all non-alphanumeric characters except hyphen
    cleaned = ''.join(c for c in text if c.isalnum() or c == '-').upper()
    
    # Common license plate patterns (add more if needed)
    patterns = [
        r'^[A-Z0-9]{2,8}$',  # 2-8 characters of letters and numbers
        r'^[A-Z]{1,3}[0-9]{1,4}$',  # 1-3 letters followed by 1-4 numbers
        r'^[0-9]{1,4}[A-Z]{1,3}$',  # 1-4 numbers followed by 1-3 letters
    ]
    
    # Check if cleaned text matches any pattern
    for pattern in patterns:
        if re.match(pattern, cleaned):
            return cleaned
            
    # If no pattern matches but we have alphanumeric characters, return them
    if len(cleaned) >= 2:  # Return if at least 2 characters
        return cleaned
        
    return None

def recognize_license_plate(image_path, debug=False):
    """
    Recognize license plate text from an image using EasyOCR with improved preprocessing
    """
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
        
        # Store original image for visualization
        original = img.copy()
        
        # Preprocess image with multiple methods
        processed_images = preprocess_image(img)
        
        if debug:
            # Show preprocessing results
            plt.figure(figsize=(15, 10))
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            titles = ['Otsu Threshold', 'Adaptive Threshold', 'CLAHE']
            for i, proc_img in enumerate(processed_images):
                plt.subplot(2, 2, i+2)
                plt.imshow(proc_img, cmap='gray')
                plt.title(titles[i])
                plt.axis('off')
            plt.show()
        
        # Collect all results from different processing methods
        all_results = []
        
        # Process original image
        original_results = reader.readtext(original)
        all_results.extend(original_results)
        
        # Process each preprocessed version
        for proc_img in processed_images:
            results = reader.readtext(proc_img)
            all_results.extend(results)
        
        if debug:
            print(f"\nAll detected text for {os.path.basename(image_path)}:")
            for idx, detection in enumerate(all_results):
                print(f"{idx+1}. Text: {detection[1]}, Confidence: {detection[2]:.2f}")
        
        # Process all detected text
        valid_detections = []
        for detection in all_results:
            text = detection[1]
            confidence = detection[2]
            
            # Extract and validate alphanumeric text
            cleaned_text = extract_alphanumeric(text)
            if cleaned_text:
                valid_detections.append((cleaned_text, confidence))
        
        # Remove duplicates while keeping highest confidence
        unique_texts = {}
        for text, conf in valid_detections:
            if text not in unique_texts or conf > unique_texts[text]:
                unique_texts[text] = conf
        
        # Sort by confidence
        final_results = sorted([(text, conf) for text, conf in unique_texts.items()],
                             key=lambda x: x[1],
                             reverse=True)
        
        if debug:
            print("\nFiltered results:")
            for text, conf in final_results:
                print(f"Text: {text}, Confidence: {conf:.2f}")
        
        # Return the best result
        if final_results:
            return final_results[0][0], final_results[0][1]
        
        # If no valid results found, try to extract any alphanumeric sequences
        partial_texts = []
        for detection in all_results:
            text = detection[1]
            # Extract any sequence of 2 or more alphanumeric characters
            alphanumeric = ''.join(c for c in text if c.isalnum()).upper()
            if len(alphanumeric) >= 2:
                partial_texts.append((alphanumeric, detection[2]))
        
        if partial_texts:
            # Sort by length and confidence
            partial_texts.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            return f"Partial: {partial_texts[0][0]}", partial_texts[0][1]
            
        return "No valid license plate detected", 0.0
            
    except Exception as e:
        return f"Error during recognition: {str(e)}", 0.0

def process_all_plates(debug=False):
    """
    Process all license plate images in the output directory
    """
    # Get all matching files
    image_files = glob.glob("output//plate_roi_*.jpg")
    
    results = []
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        plate_text, confidence = recognize_license_plate(image_path, debug=debug)
        
        # Extract name from filename
        name = os.path.basename(image_path).replace('plate_roi_', '').replace('.jpg', '')
        
        results.append({
            'name': name,
            'image_path': image_path,
            'plate_text': plate_text,
            'confidence': confidence
        })
        
        print(f"Result for {name}: {plate_text} (Confidence: {confidence:.2f})")
    
    # Print summary
    print("\nSummary of all results:")
    print("-" * 50)
    for result in results:
        print(f"Image: {result['name']}")
        print(f"Plate: {result['plate_text']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)
    
    return results

def main():
    """
    Main function to process all license plate images
    """
    print("Starting batch license plate recognition...")
    results = process_all_plates(debug=True)
    
    # Save results to file
    with open('recognition_results.txt', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Image: {result['name']}\n")
            f.write(f"Plate: {result['plate_text']}\n")
            f.write(f"Confidence: {result['confidence']:.2f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nProcessed {len(results)} images. Results saved to recognition_results.txt")

if __name__ == "__main__":
    main()