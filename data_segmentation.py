import json
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Load image and annotation
image_path = r"C:\Users\DELL\Desktop\road_harard_project\dataset\leftImg8bit\train\268\frame23013_leftImg8bit.jpg"
json_path = r"C:\Users\DELL\Desktop\road_harard_project\dataset\gtFine\train\268\frame23013_gtFine_polygons.json"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

with open(json_path, 'r') as f:
    annotation = json.load(f)

# Updated Target Labels (now including 'person' and 'truck')
target_labels = ['car', 'motorcycle', 'autorickshaw', 'vehicle fallback', 'road', 'person', 'truck']

# Blank mask initialized with uint8 type and black color (0)
selected_mask = np.zeros((height, width, 3), dtype=np.uint8)

# Fixed colors (values between 0-255)
simple_colors = {
    'road': [0, 255, 0],              # Green for road
    'car': [255, 0, 0],               # Red for car
    'motorcycle': [0, 0, 255],         # Blue for motorcycle
    'autorickshaw': [255, 255, 0],     # Yellow for autorickshaw
    'vehicle fallback': [255, 0, 255], # Pink for fallback vehicles
    'person': [255, 165, 0],           # Orange for person
    'truck': [0, 255, 255]             # Cyan for truck
}

# Store extracted objects
selected_objects = []

# Create mask for each object
for obj in annotation['objects']:
    label = obj['label']
    if label in target_labels:
        
        # Check if the polygon data is a list of points
        if isinstance(obj['polygon'], list):
            if all(isinstance(i, (int, float)) for i in obj['polygon']):
                if len(obj['polygon']) % 2 == 0:
                    polygon_points = [(obj['polygon'][i], obj['polygon'][i+1]) 
                                      for i in range(0, len(obj['polygon']), 2)]
                else:
                    print(f"Warning: Skipping malformed polygon for label {label}")
                    continue
            else:
                polygon_points = obj['polygon']

            # Create polygon as numpy array
            polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
            
            # Bounding box calculations
            x_min = int(min(polygon_points, key=lambda p: p[0])[0])
            y_min = int(min(polygon_points, key=lambda p: p[1])[1])
            x_max = int(max(polygon_points, key=lambda p: p[0])[0])
            y_max = int(max(polygon_points, key=lambda p: p[1])[1])
            
            # Fill the mask
            cv2.fillPoly(selected_mask, [polygon], color=simple_colors[label])

            # Create a bounding box around the object
            cv2.rectangle(selected_mask, (x_min, y_min), (x_max, y_max), simple_colors[label], 2)

            # Save object details
            selected_objects.append({
                "Label": label,
                "Polygon Coordinates": str(polygon_points),
                "Bounding Box": f"({x_min}, {y_min}), ({x_max}, {y_max})"
            })
        else:
            print(f"Warning: Skipping object with invalid polygon data for label: {label}")

# Save the object list to a CSV
df_selected = pd.DataFrame(selected_objects)
df_selected.to_csv('vehicle_road_objects.csv', index=False)

# Create an overlay
overlay = cv2.addWeighted(image, 0.7, selected_mask, 0.5, 0)

# Create a legend
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(selected_mask)
axs[1].set_title('Segmentation Mask')
axs[1].axis('off')

axs[2].imshow(overlay)
axs[2].set_title('Overlay Image')
axs[2].axis('off')

# Create a legend with colors
legend_elements = [plt.Line2D([0], [0], color=(color[0]/255, color[1]/255, color[2]/255), lw=4, label=label) 
                   for label, color in simple_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=len(simple_colors), fontsize=12)

plt.tight_layout()
plt.show()

print("✅ Done! Mask with Vehicles, Roads, People, and Trucks displayed.")