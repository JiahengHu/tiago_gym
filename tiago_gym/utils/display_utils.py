import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tiago_gym.utils.general_utils import run_threaded_command
import screeninfo
import cv2

class FullscreenImageDisplay:
    def __init__(self, update_interval=1, monitor_id=0):
        self.update_interval = update_interval
        monitors = screeninfo.get_monitors()
        assert monitor_id < len(monitors), "Monitor ID larger than number of connected monitors"

        self.monitor = monitors[monitor_id]
        
        run_threaded_command(self.run_mainloop)
        # self.current_image = generate_numpy_image(1920, 1080)
        self.update = False
        run_threaded_command(self.display_images)
    
    def run_mainloop(self):
        self.root = tk.Tk()
        self.root.geometry(f'{self.monitor.width}x{self.monitor.height}+0+0')
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.mainloop()
        
    def display_images(self):
        while True:
            if self.update:
                self.update = False
                if self.current_image is not None:
                    # Ensure this is done in the main thread
                    self.root.after(0, self.update_image)
            time.sleep(self.update_interval)
    
    def update_image(self):
        img = Image.fromarray(self.current_image)
        img = ImageTk.PhotoImage(img)
        self.label.config(image=img)
        self.label.image = img
        self.current_image = None
            
    def set_update(self, image):
        self.current_image = image
        self.update = True

    def get_resized_image(self, img):
        return cv2.resize(img, (self.monitor.width, self.monitor.height))
        
# Function to generate numpy images
def generate_numpy_image(width, height):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Usage example
if __name__ == "__main__":
    display1 = FullscreenImageDisplay(update_interval=1, monitor_id=0)
    # display2 = FullscreenImageDisplay(update_interval=1, monitor_id=1)

    # Simulate updates with generated numpy images
    for _ in range(5):
        time.sleep(1)
        new_image = generate_numpy_image(1920, 1080)  # Adjust size as needed
        display1.set_update(new_image)
        # display2.set_update(new_image)

