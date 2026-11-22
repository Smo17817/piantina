from pathlib import Path
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
EMBEDDINGS = BASE / "outputs" / "embeddings_extended.npz"

K_VALUES = [1, 5, 10, 15, 20]

def load(emb_path):
    dati = np.load(emb_path)
    return dati["embedding"], dati["specie"], dati["group"]

def split(emb, specie, group):
    mask_train = group == "train"
    mask_test = group == "test"
    mask_unknown = group == "unknown"

    # Il numero delle maschere deve essere uguale al numero delle istanze
    assert mask_train.sum() + mask_test.sum() + mask_unknown.sum() == len(group), "Le maschere non coprono tutte le righe"

    # Asegna direttamente tutte le istanze usando la maschera
    train_x = emb[mask_train]
    train_y = specie[mask_train]
    test_x = emb[mask_test]
    test_y = specie[mask_test]
    unknown_x = emb[mask_unknown]
    unknown_y = specie[mask_unknown]

    return train_x, train_y, test_x, test_y, unknown_x, unknown_y

def neighbours(test_x, train_x, k):
    '''
    Calcola la similarità tra tutti gli elementi di train e test.
    Vengono restituiti i k indici di train piu simili per ciascun elemento di test 
    e il valore massimo di similarità per ogni riga
    '''
    # similarità tra tutte le istanze degli embedding di train e di test
    # ci serve per identificare a quali elementi le istanze di train sono piu simili
    S = test_x @ train_x.T

    # np.argsort ordina gli indici dei vicini, dal piu grande al più piccolo (-S)
    # [:, :k] esegue uno slicing solle colonne, permettendoci di recuperare i k vicini piu prossimi
    return np.argsort(-S, axis=1)[:, :k], S.max(axis=1)

def voting(neighbors):
    return np.array([Counter(row).most_common(1)[0][0] for row in neighbors])

def predict(query_x, train_x, train_y, k):
    '''
    Calcola i k vicini più prossimi e le previsioni per query_x, unendo neighbors e voting
    '''
    nb_idx, max_scores = neighbours(query_x, train_x, k)
    labels = voting(train_y[nb_idx])

    return labels, max_scores

def check_duplicates(score, val):
    '''
    Crea una maschera per gli elementi che hanno una similarità troppo alta, potrebbero essere 
    duplicati
    '''
    return score < val

def accuracy(predictions, labels):
    '''
    Confronta tutte le labels predette con quelle reali e misura la media
    '''
    return (predictions == labels).mean()

def short_names(labels):
    '''
    Solo per la visualizzazione: "Nome comune (Nome scientifico)" -> "Nome comune".
    regex=False perché "(" in un'espressione regolare ha un significato speciale.
    '''
    return pd.Series(labels).str.split(" (", regex=False).str[0]

def confusion_matrix(real_values, predictions):
    '''
    Tabella con le specie vere sulle righe e quelle predette sulle colonne.
    Sulla diagonale le risposte corrette, fuori dalla diagonale gli errori.
    '''
    return pd.crosstab(
        short_names(real_values),
        short_names(predictions),
        rownames=["Vera"],
        colnames=["Predetta"],
        margins=True,            # aggiunge riga e colonna dei totali
        margins_name="Totale",
    )

def plot_scores(gruppi, soglia=None):
    '''
    Istogrammi sovrapposti dei punteggi (similarità del vicino più prossimo).
    gruppi: dizionario {nome: array di punteggi}
    soglia: se indicata, disegna una linea verticale in quel punto
    '''
    bordi = np.linspace(0, 1, 51)   # stessi 50 intervalli (larghi 0,02) per tutti i gruppi

    plt.figure(figsize=(10, 5))
    for nome, valori in gruppi.items():
        plt.hist(valori, bins=bordi, density=True, alpha=0.5, label=f"{nome} (n={len(valori)})") # type: ignore

    if soglia is not None:
        plt.axvline(soglia, color="black", linestyle="--", label=f"Soglia {soglia}")

    plt.xlabel("Similarità del vicino più prossimo")
    plt.ylabel("Densità")
    plt.title("Distribuzione dei punteggi")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    k = 5

    pd.set_option("display.max_columns", None)  # nessun limite al numero di colonne
    pd.set_option("display.width", None)        # adatta la larghezza al terminale

    emb, specie, group = load(EMBEDDINGS)
    train_x, train_y, test_x, test_y, unknown_x, unknown_y = split(emb, specie, group)

    predictions_test, score_test = predict(test_x, train_x, train_y, k)
    predictions_unknown, score_unknown = predict(unknown_x, train_x, train_y, k)
    
    mask = check_duplicates(score_test, 0.97)

    print(f"\n=== TEST - k: {k} ===")
    print(f"Accuracy con duplicati: {accuracy(predictions_test, test_y):.3f}")
    print(f"Accuracy senza duplicati: {accuracy(predictions_test[mask], test_y[mask]):.3f}")
    print(f"Immagini Escluse: {(~mask).sum()}")
    print(f"\nPercentili (5-50-95)")
    print(f"Immagini Classificate Correttamente: {np.percentile(score_test[predictions_test == test_y], [5, 50, 95])}")
    print(f"Immagini Classificate Erroneamente: {np.percentile(score_test[predictions_test != test_y], [5, 50, 95])}")
    print(f"Immagini Sconosciute: {np.percentile(score_unknown, [5, 50, 95])}")
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(test_y, predictions_test))

    corrette = predictions_test == test_y
    plot_scores(
        {
            "Corrette": score_test[corrette],
            "Sbagliate": score_test[~corrette],
            "Sconosciute": score_unknown,
        },
        soglia=0.7,
    )

    

