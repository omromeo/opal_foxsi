import opalfoxsi
import numpy as np
import matplotlib.pyplot as plt

import cv2



# Load the image (replace 'path_to_your_image.png' with your image path)
image_path = 'DSC_0024.png'
image = opalfoxsi.readimage(image_path)
#fig = opalfoxsi.plotimage(image,bf=10)

sz           = np.shape(image)
# Remove alpha values (Only need RGB Values)
#image        = image[:,:,0:3]
# Increase brightness to align grid due to projection issues
#image_bright = cv2.convertScaleAbs(image, 1, bf)
# Display the interactive plot
#plot = create_interactive_plot(image)






# Allow the user to select points interactively (including zoom mode)
selected_points = opalfoxsi.selectpoints(image,n=4)


# Convert the selected points to NumPy array
selected_points = np.array(selected_points)

# Extract the x and y coordinates of the selected points
xdata = selected_points[:, 0]
ydata = selected_points[:, 1]

# Print the selected coordinates (you can save them to variables or use them as needed)
print("Selected x coordinates:", xdata)
print("Selected y coordinates:", ydata)