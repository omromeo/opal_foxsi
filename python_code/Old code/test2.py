import opalfoxsi
import numpy as np
import matplotlib.pyplot as plt

import cv2
from IPython.display import display, clear_output



def create_interactive_plot(image):
        fig, ax = plt.subplots(figsize=(9,6))
        ax.imshow(image)
        ax.set_title("Select Corner Points (4) on Grid!")
        ax.axis('on')
        scatter = ax.scatter([], [], marker='o', color='red')
        # List to store selected points
        global selected_points
        selected_points = []
        def on_click(event):
            if event.xdata is not None and event.ydata is not None:
                if ax.get_navigate_mode() == 'ZOOM':
                    return  # Ignore the click if zooming is active
                x = (event.xdata)
                y = (event.ydata)
                if len(selected_points) > 3:
                    print('TOO MANY POINTS SELECTED...RETURNING!')
                    fig.canvas.mpl_disconnect(cid)
                    return  # Ignore if 4 points are already selected
                selected_points.append((x, y))  # Append the selected point to the list
                scatter.set_offsets(selected_points)  # Update the scatter plot
                print(f"Selected point: x={x}, y={y}")
        # Connect the mouse click event to the function
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        return fig
    
    
    
    
image_path = 'DSC_0024.png'
image = opalfoxsi.readimage(image_path)
# Display the interactive plot
plot = create_interactive_plot(image)