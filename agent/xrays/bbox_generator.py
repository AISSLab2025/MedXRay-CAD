import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from xrays.heatmap_generator import lateral_get_heatmap, frontal_get_heatmap

def bbox_generator(image_path: str, view: str) -> Image.Image:
    
    if view == "lateral":
       img, heatmap = lateral_get_heatmap(image_path)
    elif view == "frontal":
       img, heatmap = frontal_get_heatmap(image_path)
    else:
        return f"{view} classification not supported."  
    
    # Load the image
    # image_path = r"C:\Users\USER\Desktop\heatmap_p17229811_s53770777.jpg"
    image = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)

    # Check if the image was loaded successfully
    if image is None or image.size == 0:
        print(f"Error: Unable to load the image '{image_path}'.")
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize the grayscale image
        normalized_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(normalized_gray, (5, 5), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (two ranges to cover the full spectrum)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red regions
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.add(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the cleaned mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        min_area = 100  # Adjust this value based on your image size
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Find the contour with the largest area
        largest_contour = max(significant_contours, key=cv2.contourArea) if significant_contours else None
        
        # Draw a bounding rectangle around the largest contour
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        bbox_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(cleaned_mask, cv2.COLOR_BGR2RGB))
        # overlay_image.save("your_saved_heatmap.png")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))

        # Original Image
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('(a) Original Image')

        # # Overlay Image
        # axes[1].imshow(mask)
        # axes[1].axis('off')
        # axes[1].set_title('(b) Mask Image')
        
        axes[1].imshow(bbox_image)
        axes[1].axis('off')
        axes[1].set_title('(c) Bounding Box Image')

        plt.tight_layout()
        
        # Save or show the figure
        # Option 1: Save the final figure as an image file
        plt.savefig('bounding_box_overlay.png')

        # Option 2: Convert the figure to an Image object and return it
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        bbox_image = Image.fromarray(img_array)
        plt.close(fig)  # Close the plot to release memory
        
        return bbox_image