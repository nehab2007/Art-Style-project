from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

root = Tk()

root.title('StyleSynth')
root.geometry('900x700')
root.resizable(False, False)

def upload_image():
    # Create a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300,200))
        img_tk = ImageTk.PhotoImage(img)
        
        uploaded_img.config(image=img_tk)
        uploaded_img.image = img_tk

        # Create a Dropdown widget to select an artstyle from
        artstyle_dropdown = OptionMenu(root, selected_option, *artstyle_options)
        artstyle_dropdown.place(relx=0.5,y=440, anchor=CENTER)

        # Placeholder text
        placeholder_text = "Enter your prompt here(Optional)"

        # Create a Text widget
        text_box = Text(root, width=32, height=5, fg="grey")
        text_box.place(relx=0.36,y=480)

        # Insert the placeholder initially
        text_box.insert("1.0", placeholder_text)

        # Function to remove placeholder on focus
        def on_focus_in(event):
            if text_box.get("1.0", "end-1c") == placeholder_text:
                text_box.delete("1.0", "end")
                text_box.config(fg="black")  # Change text color

        # Function to restore placeholder if empty
        def on_focus_out(event):
            if text_box.get("1.0", "end-1c").strip() == "":
                text_box.insert("1.0", placeholder_text)
                text_box.config(fg="grey")  # Grey color for hint

        # Bind focus events
        text_box.bind("<FocusIn>", on_focus_in)
        text_box.bind("<FocusOut>", on_focus_out)

        submit_button = Button(root,text='Let the magic begin!', command='processing')
        submit_button.place(relx=0.5, y=600, anchor=CENTER)

# Add image as background
bg_image = PhotoImage(file='assets/gradient_bg16.png')
bg_label = Label(root, image = bg_image)
bg_label.place(relheight=1, relwidth=1)

# Add logo image to the window
logo = PhotoImage(file='assets/Stylesynth.png')
logo_label = Label(root, image = logo)
logo_label.place(relx=0.5, rely=0.15, anchor=CENTER)

# Add button to upload image
upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.place(relx=0.5, y=180, anchor=CENTER)

# Display uploaded image
uploaded_img = Label(root)
uploaded_img.place(x=300, y=200)

# Art style options
artstyle_options = ['Select an art style','Van Gough - Starry Night', 'Da Vinci - Mona Lisa', 'Picasso - Cubism', 'Claude Monet - Impressionism', 'Salvador Dali - Surrealism', 'Cyberpunk', 'Pixel art']

selected_option = StringVar(root)
selected_option.set(artstyle_options[0])


root.mainloop()