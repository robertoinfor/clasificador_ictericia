import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo92.h5")

class AplicacionIctericia:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador Médico de Ictericia")
        self.root.geometry("640x760")
        self.root.configure(bg="#f3f6fb")
        self.root.resizable(False, False)

        self.image = None
        self.photo = None
        self.model = None

        self.cargar_modelo()

        title_frame = tk.Frame(root, bg="#f3f6fb")
        title_frame.pack(pady=(18, 6), fill="x")

        title = tk.Label(title_frame, text="Clínica Ocular", font=("Segoe UI", 22, "bold"), bg="#f3f6fb", fg="#0c3c68")
        title.pack()

        subtitle = tk.Label(title_frame, text="Escaneo y análisis médico de la esclera para detectar ictericia", font=("Segoe UI", 11), bg="#f3f6fb", fg="#475569")
        subtitle.pack(pady=(5, 0))

        separator = ttk.Separator(root, orient="horizontal")
        separator.pack(fill="x", padx=20, pady=(16, 18))

        button_frame = tk.Frame(root, bg="#f3f6fb")
        button_frame.pack(padx=20, pady=(0, 14), fill="x")

        btn_cargar = tk.Button(button_frame, text="📁 Cargar Imagen", command=self.cargar_imagen,
                               width=18, height=2, bg="#1f6fb2", fg="white", font=("Segoe UI", 10, "bold"),
                               activebackground="#144f85", relief="raised", bd=0)
        btn_cargar.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        btn_capturar = tk.Button(button_frame, text="📷 Capturar Cámara", command=self.capturar_foto,
                                 width=18, height=2, bg="#0d8b9d", fg="white", font=("Segoe UI", 10, "bold"),
                                 activebackground="#0a6b76", relief="raised", bd=0)
        btn_capturar.grid(row=0, column=1, padx=(10, 0), sticky="ew")

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.image_frame = tk.Frame(root, bg="#ffffff", bd=1, relief="solid")
        self.image_frame.pack(padx=20, pady=5)

        self.image_canvas = tk.Canvas(self.image_frame, width=560, height=360, bg="#f8fafc", bd=0, highlightthickness=0)
        self.image_canvas.pack()
        self.image_canvas.create_text(280, 180, text="Vista previa de imagen completa", fill="#94a3b8",
                                      font=("Segoe UI", 12), tags="placeholder")

        self.result_frame = tk.Frame(root, bg="#ffffff", bd=1, relief="solid")
        self.result_frame.pack(padx=20, pady=16, fill="x")

        result_label = tk.Label(self.result_frame, text="Diagnóstico", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#0c3c68")
        result_label.pack(pady=(16, 0))

        self.result_text = tk.Label(self.result_frame, text="Cargue o capture una foto de los ojos para obtener la evaluación médica",
                                    font=("Segoe UI", 12), bg="#ffffff", fg="#334155", justify="center", wraplength=520)
        self.result_text.pack(pady=(10, 6))

        self.status_bar = tk.Label(root, text="Listo para análisis", bg="#0c3c68", fg="white", anchor="w", padx=14, font=("Segoe UI", 10))
        self.status_bar.pack(side="bottom", fill="x")
    
    def cargar_modelo(self):
        """Carga el modelo Keras en la instancia"""
        try:
            from keras.models import load_model
            from keras.layers import Dense
            import tensorflow as tf
            
            def custom_dense(**kwargs):
                kwargs.pop('quantization_config', None)
                return Dense(**kwargs)
            
            custom_objects = {'Dense': custom_dense}
            self.model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model = None

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
            
            if key == ord(' '):
                cap.release()
                cv2.destroyAllWindows()
                # Guardar temporalmente
                temp_path = "temp_foto.jpg"
                cv2.imwrite(temp_path, frame)
                self.procesar_imagen(temp_path)
                os.remove(temp_path)
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                self.status_bar.config(text="Captura cancelada")
                break
    
    def procesar_imagen(self, ruta_imagen):
        """Procesa la imagen y hace la predicción"""
        try:
            self.status_bar.config(text="Procesando imagen...")
            self.root.update()
            
            img = cv2.imread(ruta_imagen)
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                self.status_bar.config(text="Error al cargar imagen")
                return
            
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_display)
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                resample_method = getattr(Image, "LANCZOS", Image.BICUBIC)
            img_pil.thumbnail((560, 360), resample_method)
            self.photo = ImageTk.PhotoImage(img_pil)
            self.image_canvas.delete("all")
            self.image_canvas.create_rectangle(0, 0, 560, 360, outline="#cbd5e1")
            self.image_canvas.create_image(280, 180, image=self.photo, anchor="center")

            if self.model is None:
                self.result_text.config(text="❌ Modelo no cargado correctamente", fg="red")
                self.status_bar.config(text="Error: Modelo no disponible")
                return
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_modelo = cv2.resize(img_rgb, (224, 224)).astype('float32') / 255.0
            img_array = np.expand_dims(img_modelo, axis=0)
            
            prediccion = self.model.predict(img_array, verbose=0)
            probabilidad = 1 - prediccion[0][0]
            
            print(f"DEBUG - Valor predicción: {probabilidad:.4f}")
            
            if probabilidad > 0.6:
                resultado = "⚠️ ICTERICIA DETECTADA"
                color = "red"
                confianza = f"{probabilidad*100:.2f}%"
            else:
                resultado = "✅ SIN ICTERICIA"
                color = "green"
                confianza = f"{(1-probabilidad)*100:.2f}%"
            
            self.result_text.config(text=f"{resultado}\nConfianza: {confianza}", fg=color, font=("Arial", 12, "bold"))
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
