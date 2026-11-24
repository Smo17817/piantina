from embedding import build_transform, load_model, check_tensor, denormalize, embed_batch, IMAGE_SIZE, RESIZE_SIZE
from knn import load, split, predict, THRESHOLD, K, EMBEDDINGS
from PIL import Image
import cv2
import torch
import numpy as np
import time
from collections import deque, Counter

WINDOW_NAME = "Piantina"
WINDOW_SIZE = 15

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

def crop_box(larghezza, altezza):
    '''
    Trova l'area osservata dal modello nell'immagine originale.
    Viene restituito il vertice in alto a sinistra del quadrato.
    '''
    scale_factor = RESIZE_SIZE / min([larghezza, altezza])
    square_side = int(IMAGE_SIZE / scale_factor)

    v1 = ((larghezza - square_side) // 2, (altezza - square_side) // 2)
    v2 = (v1[0] + square_side, v1[1] + square_side)

    return v1, v2

def draw_label(img, testo, pos, colore, scala=0.7, spessore=2, bordo=2):
    '''Testo colorato con contorno bianco, ottenuto da copie spostate del testo.'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    for dx in (-bordo, 0, bordo):
        for dy in (-bordo, 0, bordo):
            if dx or dy:   # salta (0, 0): è la posizione del testo colorato
                cv2.putText(img, testo, (x + dx, y + dy), font, scala, (255, 255, 255), spessore, cv2.LINE_AA)
    cv2.putText(img, testo, (x, y), font, scala, colore, spessore, cv2.LINE_AA)

def stable_result(specie_q, punteggi_q, soglia):
    aggr_score = np.mean(punteggi_q)
    if aggr_score >= soglia:
        return Counter(specie_q).most_common(1)[0][0], aggr_score
    return "Unknown", aggr_score

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

        larghezza = frame.shape[1]
        altezza = frame.shape[0]
        v1, v2 = crop_box(larghezza, altezza)

        # Predizioni più stabili su WINDOW_SIZE elementi
        specie_q = deque(maxlen=WINDOW_SIZE)
        punteggi_q = deque(maxlen=WINDOW_SIZE)

        # FPS
        t_prec = time.perf_counter()

        while True:
            # legge il frame, e restituisce True (se la lettura è corretta), (altezza, larghezza, 3)
            ok, frame = cap.read()
            
            if not ok:
                raise RuntimeError("Errore durante la lettura del frame")

            # Copia del frame che ci serve solo per disegnare
            display = frame.copy()

            # Aggiornamento FPS
            t = time.perf_counter()
            fps = 1 / (t - t_prec)
            t_prec = t
            draw_label(display, f"FPS: {fps:.2f}", (25, altezza - 20), (0, 0, 0), 0.7, 2, 1)

            t_frame = frame_to_tensor(frame, transform)
            emb = embed_batch(model, t_frame, device).numpy()
            label, score = predict(emb, galleria, galleria_labels, K)
            specie_q.append(label[0].split(" (")[0])
            punteggi_q.append(score[0])

            lab, scr = stable_result(specie_q, punteggi_q, THRESHOLD)   
            if not lab == "Unknown":
                cv2.rectangle(display, v1, v2, (0, 255, 0), 2) 
                draw_label(display, f"{lab}: {scr:.2f}", (25, 25), (0, 255, 0), 0.7, 2, 1)
            else:
                cv2.rectangle(display, v1, v2, (0, 0, 255), 2)
                draw_label(display, f"Non riconosciuta: {scr:.2f}", (25, 25), (0, 0, 255), 0.7, 2, 1)
            
            # Mostra il frame, con nome "Piantina"
            cv2.imshow(WINDOW_NAME, display)
            tasto = cv2.waitKey(1) & 0xFF

            # Chiusura con tasto q
            if tasto == ord("q"):
                break
            # Chiusura con X della finestra
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()