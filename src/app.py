import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from style_transfer import apply_style

# --- Initialize main window ---
root = tk.Tk()
root.title("AI Art Style Transfer")
root.geometry("1200x700")
root.configure(bg="#1E1E1E")

# --- Global Variables ---
selected_image_path = None
output_img_tk = None

predefined_styles = [
    "Van Gogh - Starry Night", "Da Vinci - Mona Lisa", "Picasso - Cubism",
    "Claude Monet - Impressionism", "Salvador Dali - Surrealism", "Cyberpunk", "Custom"
]

# --- Load Default Images for UI ---
default_img = Image.new("RGB", (300, 300), color=(50, 50, 50))
default_img_tk = ImageTk.PhotoImage(default_img)

# --- Select Image Function ---
def select_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        selected_image_path = file_path
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        preview_label.config(image=img_tk)
        preview_label.image = img_tk

# --- Process Image Function ---
def process_image():
    global selected_image_path, output_img_tk

    if not selected_image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return

    selected_style = style_var.get()
    user_text = user_input.get().strip()

    if selected_style == "Custom" and not user_text:
        messagebox.showerror("Error", "Please enter a custom style prompt.")
        return

    user_prompt = user_text if selected_style == "Custom" else user_text
    final_style = selected_style if selected_style != "Custom" else user_text

    try:
        output_path = apply_style(selected_image_path, final_style, user_prompt)

        # Load and display styled image
        styled_img = Image.open(output_path).resize((300, 300))
        output_img_tk = ImageTk.PhotoImage(styled_img)
        output_label.config(image=output_img_tk)
        output_label.image = output_img_tk

        messagebox.showinfo("Success", f"Style applied: {final_style}\nSaved to: {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply style:\n{e}")

# --- UI Layout ---
frame = tk.Frame(root, bg="#2A2A2A", padx=20, pady=20, relief="solid", bd=2)
frame.pack(pady=20)

tk.Label(frame, text="AI Art Style Transfer", font=("Helvetica", 24, "bold"), fg="#D4D4D4", bg="#2A2A2A").grid(row=0, column=0, columnspan=2, pady=10)

# Image Previews
preview_label = tk.Label(frame, image=default_img_tk, bg="#3A3A3A", relief="ridge", bd=2)
preview_label.grid(row=1, column=0, padx=15, pady=10)

output_label = tk.Label(frame, image=default_img_tk, bg="#3A3A3A", relief="ridge", bd=2)
output_label.grid(row=1, column=1, padx=15, pady=10)

# Buttons and Inputs
tk.Button(frame, text="Select Image", font=("Helvetica", 14), command=select_image, bg="#0078D4", fg="white", padx=15, pady=8, relief="flat").grid(row=2, column=0, columnspan=2, pady=10)

style_var = tk.StringVar()
style_var.set(predefined_styles[0])
ttk.Combobox(frame, textvariable=style_var, values=predefined_styles, font=("Helvetica", 12), state="readonly", width=40).grid(row=3, column=0, columnspan=2, pady=10)

user_input = tk.Entry(frame, font=("Helvetica", 12), width=45, relief="solid", bd=1, fg="#D4D4D4", bg="#3A3A3A")
user_input.insert(0, "Describe the style you want...")
user_input.grid(row=4, column=0, columnspan=2, pady=10)

tk.Button(frame, text="Apply Style", font=("Helvetica", 14), command=process_image, bg="#28A745", fg="white", padx=15, pady=8, relief="flat").grid(row=5, column=0, columnspan=2, pady=20)

# --- Start Main Loop ---
root.mainloop()
