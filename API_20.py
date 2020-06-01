from keras.engine.saving import load_model
import numpy as np
import cv2
from tkinter import Tk, Label, Scale, VERTICAL
from PIL import Image, ImageTk


class API:
    def __init__(self, master):
        self.master = master
        master.title("Face Creation")

        self.model = load_model('run\\weights\\decoder_VAE_faces_20_dim.h5')

        self.labels = ["Edad", "Sonrisa", "Ojos", "Redondez", "Dientes", "Genero", "Edad", "NaN", "Raza", "Tono piel",
                       "Calidez", "Angulo cabeza", "Edad", "Color pelo", "Nan", "Nan", "Tono piel", "Genero", "Nan", "Luz"]

        self.image_size = (256, 256)
        self.image = ImageTk.PhotoImage(Image.fromarray(np.zeros((self.image_size[0], self.image_size[1], 3), 'uint8')))
        self.canvas = Label(master, image=self.image)
        self.canvas.grid(column=0, row=0)

        self.scales = []
        for i in [2, 4]:
            for j in range(10):
                self.label = Label(master, text=self.labels[j if i == 2 else (j + 10)])
                self.label.grid(column=j+2, row=i-1)
                self.scales.append(Scale(master, from_=-50, to=50, orient=VERTICAL, command=self.updateValue))
                self.scales[-1].grid(column=j+2, row=i)

    def updateValue(self, event):
        input_tensor = []
        for scale in self.scales:
            input_tensor.append(scale.get() / 10)
        input_tensor = np.array(input_tensor, 'float32').reshape((1, 20))
        output_tensor = self.model.predict(input_tensor)[0]
        output_image = (output_tensor * 255).astype('uint8')
        output_image = cv2.resize(output_image, self.image_size, cv2.INTER_CUBIC)

        image = ImageTk.PhotoImage(Image.fromarray(output_image))

        self.canvas.configure(image=image)
        self.canvas.image = image


root = Tk()
my_gui = API(root)
root.mainloop()
