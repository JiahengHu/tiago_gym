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

class FullscreenStringDisplay:
    def __init__(self, update_interval=1, monitor_id=0):
        self.update_interval = update_interval
        monitors = screeninfo.get_monitors()
        assert monitor_id < len(monitors), "Monitor ID larger than number of connected monitors"

        self.monitor = monitors[monitor_id]
        
        run_threaded_command(self.run_mainloop)
        # self.current_image = generate_numpy_image(1920, 1080)
        # self.update = False
        # run_threaded_command(self.display_images)
    
    def run_mainloop(self):
        self.root = tk.Tk()
        self.root.geometry(f'{self.monitor.width}x{self.monitor.height}+0+0')
        self.label = tk.Label(self.root, font=('calibri', 300, 'bold'), background='black', foreground='white')
        # self.label.pack(pady=50, padx=50)
        self.label.pack()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.mainloop()
        
    def display_text(self, text, relx=None, rely=None):
        self.label.config(text=text)
        if relx is None:
            relx = np.random.random()*0.4 + 0.1
        if rely is None:
            rely = np.random.random()*0.7 + 0.1
        self.label.place(relx=relx, rely=rely)
        
# Function to generate numpy images
def generate_numpy_image(width, height):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Usage example
if __name__ == "__main__":
    display1 = FullscreenStringDisplay(update_interval=1, monitor_id=0)

    # Simulate updates with generated numpy images
    for x in range(11):
        time.sleep(1)
        display1.display_text(text="8:00 PM", rely=x/10, relx=0.5)
    input('enter')


    # import tkinter as tk
    # from time import strftime

    # # Function to update the digital clock label
    # def update_time():
    #     string_time = strftime('%H:%M:%S %p')
    #     digital_clock.config(text=string_time)
    #     digital_clock.after(1000, update_time)

    # # Main Tkinter window
    # root = tk.Tk()
    # root.title("Digital Clock")

    # # Digital clock label configuration
    # digital_clock = tk.Label(root, font=('calibri', 40, 'bold'), background='black', foreground='white')
    # digital_clock.pack(pady=20)

    # # Initial call to update_time function
    # update_time()

    # Tkinter main loop
    # root.mainloop()
