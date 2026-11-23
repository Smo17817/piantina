from embedding import build_transform, load_model, check_tensor, denormalize, embed_batch
from knn import load, split, predict, THRESHOLD, K, EMBEDDINGS
from PIL import Image
import cv2
import torch
import numpy as np

def frame_to_tensor(frame, transform):
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_frame)

    return transformed_frame.unsqueeze(0)

def show_frame(tensor):
    '''Mostra il tensore (1, 3, 224, 224) così come lo vede il modello.'''
    img = denormalize(tensor[0])                 # [0] toglie il batch → (3, 224, 224), valori 0–1
    img = img.permute(1, 2, 0).numpy()           # canali in fondo → (224, 224, 3), array NumPy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # OpenCV vuole BGR
    cv2.imshow("Input modello", img)


if __name__ == "__main__":
    # Assegna a cap il primo dispositivo webcam (0), usando DirectShow di Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Webcam non disponibile")
    
    try:
        ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Errore durante la lettura del frame")

        print(f"Risoluzione Webcam: {frame.shape[1]} x {frame.shape[0]}")

        # Caricamenti da embeddings
        transform = build_transform()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(device)

        # Caricamneti da knn
        emb, specie, group = load(EMBEDDINGS)
        train_x, train_y, test_x, test_y,_,_ = split(emb, specie, group)
        galleria = np.concatenate([train_x, test_x])
        galleria_labels = np.concatenate([train_y, test_y])

        while True:
            # legge il frame, e restituisce True (se la lettura è corretta), (altezza, larghezza, 3)
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Errore durante la lettura del frame")

            # Mostra il frame, con nome "Piantina"
            cv2.imshow("Piantina", frame)
            tasto = cv2.waitKey(1) & 0xFF

            if tasto == ord("c"):
                t_frame = frame_to_tensor(frame, transform)
                emb = embed_batch(model, t_frame, device).numpy()
                label, score = predict(emb, galleria, galleria_labels, K)

                if score > THRESHOLD:
                    print(f"Risultato: {label}")

                print(f"Score: {score}")

            # Chiusura con tasto q
            if tasto == ord("q"):
                break
            # Chiusura con X della finestra
            if cv2.getWindowProperty("Piantina", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()