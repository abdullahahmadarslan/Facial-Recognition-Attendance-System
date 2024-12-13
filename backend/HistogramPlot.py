from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = './histogram_test/histo.png'  
image = Image.open(image_path)

# Convert to grayscale
gray_image = image.convert('L')

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.title("Grayscale Image")
# plt.show()

# Convert to NumPy array and flatten
pixel_values = np.array(gray_image).flatten()

# Plot the histogram
plt.hist(pixel_values, bins=256, range=(0, 255), color='gray', edgecolor='black')
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
