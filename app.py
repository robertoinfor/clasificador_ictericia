import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Cargar modelo con la instalación standalone de Keras
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo84.h5")
model = None

def cargar_modelo():
    global model
    try:
        from keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        model = None

cargar_modelo()

class AplicacionIctericia:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Ictericia en Ojos")
        self.root.geometry("600x700")
        self.root.configure(bg="lightgray")
        
        # Variables
        self.image = None
        self.photo = None
        
        # Título
        title = tk.Label(root, text="Clasificador de Ictericia", font=("Arial", 18, "bold"), bg="lightgray")
        title.pack(pady=10)
        
        # Frame para botones
        button_frame = tk.Frame(root, bg="lightgray")
        button_frame.pack(pady=10)
        
        # Botón cargar imagen
        btn_cargar = tk.Button(button_frame, text="📁 Cargar Foto", command=self.cargar_imagen, 
                               width=20, height=2, bg="blue", fg="white", font=("Arial", 10, "bold"))
        btn_cargar.grid(row=0, column=0, padx=5)
        
        # Botón capturar foto
        btn_capturar = tk.Button(button_frame, text="📷 Tomar Foto", command=self.capturar_foto,
                                width=20, height=2, bg="green", fg="white", font=("Arial", 10, "bold"))
        btn_capturar.grid(row=0, column=1, padx=5)
        
        # Area de imagen
        self.image_label = tk.Label(root, bg="white", width=100, height=20)
        self.image_label.pack(pady=10)
        
        # Area de resultado
        self.result_frame = tk.Frame(root, bg="white", relief="sunken", bd=2)
        self.result_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.result_text = tk.Label(self.result_frame, text="Cargue o capture una foto de los ojos", 
                                   font=("Arial", 12), bg="white", justify="center", wraplength=500)
        self.result_text.pack(pady=20)
        
        # Barra de estado
        self.status_bar = tk.Label(root, text="Listo", bg="gray", fg="white", justify="left")
        self.status_bar.pack(side="bottom", fill="x")
    
    def cargar_imagen(self):
        """Carga una imagen desde el disco"""
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if archivo:
            self.procesar_imagen(archivo)
    
    def capturar_foto(self):
        """Captura foto con la cámara"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara")
            return
        
        self.status_bar.config(text="Abriendo cámara... Presione SPACE para capturar, ESC para cancelar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Captura de Foto - Presione ESPACIO para capturar, ESC para cancelar", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # ESPACIO
                cap.release()
                cv2.destroyAllWindows()
                # Guardar temporalmente
                temp_path = "temp_foto.jpg"
                cv2.imwrite(temp_path, frame)
                self.procesar_imagen(temp_path)
                os.remove(temp_path)
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                self.status_bar.config(text="Captura cancelada")
                break
    
    def procesar_imagen(self, ruta_imagen):
        """Procesa la imagen y hace la predicción"""
        try:
            self.status_bar.config(text="Procesando imagen...")
            self.root.update()
            
            # Cargar imagen
            img = cv2.imread(ruta_imagen)
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                self.status_bar.config(text="Error al cargar imagen")
                return
            
            # Redimensionar para mostrar
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_display)
            img_pil.thumbnail((400, 400))
            self.photo = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.photo)
            
            # Predecir
            if model is None:
                self.result_text.config(text="❌ Modelo no cargado correctamente", fg="red")
                self.status_bar.config(text="Error: Modelo no disponible")
                return
            
            # Preparar imagen para modelo (redimensionar a 224x224 o el tamaño esperado)
            img_modelo = cv2.resize(img, (224, 224)).astype('float32') / 255.0
            img_array = np.expand_dims(img_modelo, axis=0)
            
            # Hacer predicción
            prediccion = model.predict(img_array, verbose=0)
            probabilidad = prediccion[0][0]
            
            # Mostrar resultado
            if probabilidad > 0.5:
                resultado = "⚠️ ICTERICIA DETECTADA"
                color = "red"
                confianza = f"{probabilidad*100:.2f}%"
            else:
                resultado = "✅ SIN ICTERICIA"
                color = "green"
                confianza = f"{(1-probabilidad)*100:.2f}%"
            
            self.result_text.config(text=f"{resultado}\nConfianza: {confianza}", fg=color, font=("Arial", 14, "bold"))
            self.status_bar.config(text="Predicción completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error procesando imagen: {e}")
            self.status_bar.config(text="Error en procesamiento")

def main():
    if not os.path.exists(MODEL_PATH):
        root = tk.Tk()
        messagebox.showerror("Error", f"Modelo no encontrado en:\n{MODEL_PATH}\n\nPor favor, asegúrese de que modelo84.h5 esté en la carpeta de la aplicación")
        root.destroy()
        return
    
    root = tk.Tk()
    app = AplicacionIctericia(root)
    root.mainloop()

if __name__ == "__main__":
    main()
