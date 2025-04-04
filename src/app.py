import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from style_transfer import apply_style

# Initialize main window
root = tk.Tk()
root.title("AI Art Style Transfer")
root.geometry("1200x700")
root.configure(bg="#1E1E1E")

# Load default images
default_img = Image.new("RGB", (300, 300), color=(50, 50, 50))
default_img_tk = ImageTk.PhotoImage(default_img)

default_output_img = Image.new("RGB", (300, 300), color=(30, 30, 30))
default_output_img_tk = ImageTk.PhotoImage(default_output_img)

# File path variable
selected_image_path = None
predefined_styles = ["Van Gogh - Starry Night", "Da Vinci - Mona Lisa", "Picasso - Cubism", 
                     "Claude Monet - Impressionism", "Salvador Dali - Surrealism", "Cyberpunk"]

def select_image():
    global selected_image_path, preview_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image_path = file_path
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        preview_label.config(image=img_tk)
        preview_label.image = img_tk

def process_image():
    global selected_image_path, user_input, style_var
    if not selected_image_path:
        messagebox.showerror("Error", "Please select an image first")
        return

    selected_style = style_var.get()
    user_text = user_input.get()
    user_prompt = user_input.get()

    if selected_style == "Custom" and not user_text:
        messagebox.showerror("Error", "Please enter a style description")
        return

    applied_style = user_text if selected_style == "Custom" else selected_style
    styled_image = apply_style(selected_image_path, applied_style, user_prompt)

    if isinstance(styled_image, Image.Image):
        # Save & Display result
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "styled_image.jpg")
        styled_image.save(output_path)
        styled_img_tk = ImageTk.PhotoImage(styled_image.resize((300, 300)))
        output_label.config(image=styled_img_tk)
        output_label.image = styled_img_tk
        messagebox.showinfo("Success", f"Style Applied: {applied_style}\nImage saved to {output_path}")
    else:
        messagebox.showerror("Error", "Failed to apply style transfer")

# UI Elements
frame = tk.Frame(root, bg="#2A2A2A", padx=20, pady=20, relief="solid", bd=2)
frame.pack(pady=20)

title = tk.Label(frame, text="AI Art Style Transfer", font=("Helvetica", 24, "bold"), fg="#D4D4D4", bg="#2A2A2A")
title.grid(row=0, column=0, columnspan=2, pady=10)

# Image display labels
preview_label = tk.Label(frame, image=default_img_tk, bg="#3A3A3A", relief="ridge", bd=2)
preview_label.grid(row=1, column=0, padx=15, pady=10)

output_label = tk.Label(frame, image=default_output_img_tk, bg="#3A3A3A", relief="ridge", bd=2)
output_label.grid(row=1, column=1, padx=15, pady=10)

btn_select = tk.Button(frame, text="Select Image", font=("Helvetica", 14), command=select_image, bg="#0078D4", fg="white", padx=15, pady=8, relief="flat", cursor="hand2")
btn_select.grid(row=2, column=0, columnspan=2, pady=10)

# Dropdown for predefined styles
style_var = tk.StringVar()
style_var.set(predefined_styles[0])
style_menu = ttk.Combobox(frame, textvariable=style_var, values=predefined_styles + ["Custom"], font=("Helvetica", 12), state="readonly", width=40)
style_menu.grid(row=3, column=0, columnspan=2, pady=10)

# Text input for custom styles
user_input = tk.Entry(frame, font=("Helvetica", 12), width=45, relief="solid", bd=1, fg="#D4D4D4", bg="#3A3A3A")
user_input.grid(row=4, column=0, columnspan=2, pady=10)
user_input.insert(0, "Describe the style you want...")

btn_process = tk.Button(frame, text="Apply Style", font=("Helvetica", 14), command=process_image, bg="#28A745", fg="white", padx=15, pady=8, relief="flat", cursor="hand2")
btn_process.grid(row=5, column=0, columnspan=2, pady=20)

root.mainloop()
