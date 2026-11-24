# Piantina 🪴

Riconoscimento di piante domestiche in tempo reale tramite webcam.

Ogni frame viene trasformato in un embedding da **DINOv2** e classificato con un **k-NN** rispetto a una galleria di immagini di riferimento. Il risultato mostrato è stabilizzato sugli ultimi frame: la specie è quella più votata nella finestra, e se la similarità media è sotto una soglia la pianta viene indicata come **non riconosciuta**.

Il modello non viene addestrato: la "conoscenza" del sistema è la galleria stessa.

## Come funziona

```
frame webcam → ritaglio centrale 224×224 → DINOv2 (token CLS, normalizzato L2)
             → k-NN sulla galleria (similarità coseno, k = 5)
             → finestra degli ultimi 15 frame → specie più votata + similarità media
             → similarità media ≥ soglia ? specie : "Non riconosciuta"
```

Il modello vede solo il **quadrato centrale** dell'inquadratura, indicato sul video da un riquadro: la pianta va tenuta al suo interno.

## Dataset

[*House Plant Species*](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species): 14.790 immagini di 47 specie, raccolte da Bing Immagini e curate a mano.

Il progetto usa 8 specie:

`Aloe Vera`, `Snake plant (Sanseviera)`, `Jade plant (Crassula ovata)`, `Monstera Deliciosa (Monstera deliciosa)`, `Pothos (Ivy arum)`, `ZZ Plant (Zamioculcas zamiifolia)`, `Calathea`, `Prayer Plant (Maranta leuconeura)`

Le immagini delle 8 specie sono divise in **train** (80%, la galleria) e **test** (20%), in modo stratificato e con seed fisso. Altre 20 immagini per ciascuna delle 39 specie restanti formano il gruppo **unknown**, usato per scegliere la soglia di riconoscimento. La divisione è salvata nei CSV di `split/`.

## Installazione

Requisiti: Python 3.10 e una GPU NVIDIA con driver che supporti CUDA 13.2 (funziona anche su CPU, più lentamente).

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash su Windows
python -m pip install -r requirements.txt
```

`requirements.txt` installa PyTorch con CUDA 13.2 dall'indice di pytorch.org. **Senza GPU NVIDIA**, sostituisci la prima riga con `--extra-index-url https://download.pytorch.org/whl/cpu` e togli `+cu132` dalle versioni di `torch` e `torchvision`.

Il dataset non è incluso nel repository: scaricalo da Kaggle ed estrailo in modo da avere

```
dataset/house_plant_species/<specie>/<immagini>
```

## Utilizzo

Gli script vanno eseguiti in quest'ordine; ognuno legge i file prodotti dal precedente.

| Script | Cosa fa | Produce |
|---|---|---|
| `eda.py` | Esplora il dataset (conteggi, estensioni, immagini corrotte, formati) e crea la divisione train/test/unknown | `split/extended_split.csv` |
| `embedding.py` | Calcola con DINOv2 (`facebook/dinov2-small`) l'embedding di ogni immagine del CSV | `outputs/embeddings_extended.npz` |
| `knn.py` | Valuta il k-NN sul test: accuratezza, matrice di confusione, distribuzione delle similarità e tabella delle soglie | — |
| `webcam.py` | Classificazione in tempo reale dalla webcam | — |

```bash
python eda.py
python embedding.py
python knn.py
python webcam.py
```

I CSV di `split/` sono già nel repository: per riprodurre i risultati basta partire da `embedding.py`.

**Comandi della finestra webcam:** `q` (o la X della finestra) per uscire.

### Parametri principali

| Parametro | File | Valore |
|---|---|---|
| `K` (vicini del k-NN) | `knn.py` | 5 |
| `THRESHOLD` (soglia di riconoscimento) | `knn.py` | 0.60 |
| `WINDOW_SIZE` (frame per la stabilizzazione) | `webcam.py` | 15 |

## Risultati

Sul test (592 immagini delle 8 specie), con k = 5:

- **accuratezza: 97,8%** (13 errori);
- 9 errori su 13 sono scambi tra **Calathea** e **Prayer Plant**, due specie della stessa famiglia e molto simili;
- i 16 quasi-duplicati tra train e test non influenzano il risultato (97,7% escludendoli).

Con la soglia a 0,60 il sistema risponde al 99,7% delle immagini di test e rifiuta circa metà delle piante sconosciute del dataset. Gli oggetti non vegetali inquadrati dalla webcam (persone, mobili, pareti) hanno similarità molto più basse e vengono rifiutati.

## Limiti

- **Domain shift:** le immagini del dataset sono foto del web, ben inquadrate. Dalla webcam le similarità sono più basse (circa 0,72–0,79 contro 0,88 del test), soprattutto con poca luce.
- **Piante sconosciute simili:** una pianta non compresa nelle 8 specie ma somigliante a una di esse può superare la soglia ed essere classificata come quella specie.
- **Calathea e Prayer Plant** vengono confuse tra loro: la soglia non aiuta, perché gli errori hanno similarità alte.
