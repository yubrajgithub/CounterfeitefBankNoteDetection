import cv2

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize image to 640x480
    resized_image = cv2.resize(image, (640, 480))
    
    return resized_image

# Example usage:
input_image_path = r"C:\Users\yubra\Downloads\IMG_20190327_130134.jpg"
processed_image = preprocess_image(input_image_path)
