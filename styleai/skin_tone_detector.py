import cv2
import numpy as np

def detect_skin_tone(image_path):
    """
    Detect skin tone category from uploaded image.
    Returns: Fair, Medium, Olive, or Deep
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return "Invalid Image"

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for faster processing
    image = cv2.resize(image, (200, 200))

    # Calculate average color
    avg_color = np.mean(image, axis=(0, 1))

    r, g, b = avg_color

    # Calculate brightness
    brightness = (r + g + b) / 3

    # Simple classification
    if brightness > 200:
        skin_tone = "Fair"
    elif brightness > 160:
        skin_tone = "Medium"
    elif brightness > 120:
        skin_tone = "Olive"
    else:
        skin_tone = "Deep"

    return skin_tone


# Test the function
if __name__ == "__main__":
    img_path = "test.jpg"
    tone = detect_skin_tone(img_path)
    print("Detected Skin Tone:", tone)
    