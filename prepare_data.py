import glob

import pandas as pd
import r2pipe
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

CLASSES = [
    "APT28",
    "APT29",
    "Energetic Bear",
    "APT3",
    "Lazarus",
    "Equation Group",
    "APT19",
    "Turla",
    "APT30",
    "Sandworm",
    "FIN7",
    "Ke3chang",
    "Lotus Blossom",
    "Molerats",
    "Magic Hound",
    "Desert Falcons",
    "Bronze Butler",
    "Emissary Panda",
    "Transparent Tribe",
    "APT34",
    "Volatile Cedar",
    "APT10",
    "TA505",
    "Gamaredon",
    "APT32",
    "Patchwork",
]


def extract_opcodes(file):
    r2 = r2pipe.open(file)
    r2.cmd("aaa")
    instructions = r2.cmd("pd 1000").split("\n")
    opcodes = [line.split()[1] for line in instructions if len(line.split()) > 1]
    return " ".join(opcodes)


def prepare_data():
    texts = []
    labels = []
    for c in CLASSES:
        files = glob.glob(f"Data/source/{c}/*.asm")
        for file in tqdm.tqdm(
            files, desc=f"[INFO] Extracting Features From Class :: {c}"
        ):
            texts.append(extract_opcodes(file))
            labels.append(c)
    cv = TfidfVectorizer(ngram_range=(5, 5), max_features=2048)
    cv.fit(texts)
    features = cv.transform(texts)
    feature_names = cv.get_feature_names_out()
    features = features.toarray()
    df = pd.DataFrame(features, columns=feature_names)
    df["Class"] = labels
    df.to_csv("Data/data.csv", index=False)
    print('Feature Shape', df.shape)


if __name__ == "__main__":
    prepare_data()
