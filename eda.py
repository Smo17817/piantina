from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import pandas as pd

ESTENSIONI = {".jpe", ".jpg", ".png", ".jfif", ".jpeg", ".webp", ".gif"}
SUBSET = {
    "Aloe Vera", 
    "Snake plant (Sanseviera)", 
    "Jade plant (Crassula ovata)", 
    "Monstera Deliciosa (Monstera deliciosa)", 
    "Pothos (Ivy arum)", 
    "ZZ Plant (Zamioculcas zamiifolia)", 
    "Calathea", 
    "Prayer Plant (Maranta leuconeura)"
    }

FORMATO_ATTESO = {
    ".jpg":  "JPEG",
    ".jpeg": "JPEG",
    ".jpe":  "JPEG",
    ".jfif": "JPEG",
    ".png":  "PNG",
    ".webp": "WEBP",
    ".gif":  "GIF",
}

SEED = 42

BASE = Path(__file__).resolve().parent
root = BASE / "dataset" / "house_plant_species"

def ext_count(plant_index):
    # Conteggio delle estensioni
    ext = Counter(img.suffix.lower() for img,_ in plant_index)

    for e, c in ext.most_common():
        print(f"{e:<5}: {c:>5}")

def img_count(plant_index):
    # Conteggio delle immagini per ogni specie
    conteggi = Counter(specie for _, specie in plant_index)

    max_length = max(len(nome) for nome in conteggi)
    for nome, n in conteggi.most_common():
        print(f"{nome:<{max_length}}: {n:>5}")

    print(f"Totale: {sum(conteggi.values())}")

def img_ops(plant_index):
    '''
    Si verifica che le immagini non siano corrotte ed altre informazioni utili
    '''
    success = []
    fail = []

    for img_path,_ in plant_index:
        try:
            with Image.open(img_path) as img:
                img.load()
                success.append((img_path, img.format, img.mode, img.size))
        except UnidentifiedImageError as e:
            fail.append((img_path, type(e).__name__))
            print(f"ERRORE immagine {img_path}\n {e}")
        except OSError as e:
            fail.append((img_path, type(e).__name__))
            print(f"ERRORE immagine {img_path}\n {e}")

    # Stampa immagini corrotte
    if len(fail) > 0:
        print(f"Totale immagini corrotte: {len(fail)}")
        for img, err in fail:
            print(f"{img}: {err}")

    # Conteggio modalità (RGB, RGBA...)
    print(Counter(mode for _,_,mode,_ in success))

    # Controllo formati
    print(Counter(fmt for path,fmt,_,_ in success if FORMATO_ATTESO[path.suffix.lower()] != fmt))

    # Stampa dimensioni minime
    print(f"Larghezza Minima: {min(size[0] for _,_,_,size in success)}")
    print(f"Altezza Minima: {min(size[1] for _,_,_,size in success)}")

# --- Creazione .csv con split ---
def csv_split(plant_index, file_name="split.csv"):
    df_known = pd.DataFrame([(path.relative_to(root).as_posix(), specie) for path, specie in plant_index if specie in SUBSET], columns=["path", "specie"])
    test = df_known.groupby("specie").sample(frac=0.2, random_state=SEED)

    df_known["group"] = "train"
    df_known.loc[test.index, "group"] = "test"

    # Aggiunta immagini sconosciute per determinare il comportamento del modello in caso di indecisione
    df_unknown = pd.DataFrame([(path.relative_to(root).as_posix(), specie, "unknown") for path, specie in plant_index if specie not in SUBSET], columns=["path", "specie", 'group'])
    df_unknown_subsample = df_unknown.groupby("specie").sample(n=20, random_state=SEED)

    df = pd.concat([df_known, df_unknown_subsample], ignore_index=True)

    df.to_csv(BASE / file_name, index=False)

def check_split(file_name="split.csv"):
    df = pd.read_csv(BASE / file_name)
    print(pd.crosstab(df["specie"], df["group"]))

if __name__ == "__main__":
    # Indice strutturato come segue (Path di ogni immagine: Specie)
    plant_index = sorted([(img, img.parent.name) for img in root.rglob('*') if img.is_file() and img.suffix.lower() in ESTENSIONI])
    
    # img_ops(plant_index)
    csv_split(plant_index, "extended_split.csv")
    check_split("extended_split.csv")