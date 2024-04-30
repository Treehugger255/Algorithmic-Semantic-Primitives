import json
import os
from tqdm import tqdm
import argparse
from collections import defaultdict

# NOTE: Only use the words in Russian tagged as having pos adj, adv, noun, or verb, since these are considered "lexical"
# See the Primes one for more details
FILES = [
    "kaikki.org-dictionary-Русский-by-pos-adj.json", 
    "kaikki.org-dictionary-Русский-by-pos-adv.json", 
    "kaikki.org-dictionary-Русский-by-pos-noun.json", 
    "kaikki.org-dictionary-Русский-by-pos-verb.json"
]

def build_dict(files, kostiuk_format=False):
    dic: dict[str, list[str]] = defaultdict(list)
    for file in files:
        with open(file, encoding="utf-8") as f:
            # Each line contains a separate json record, so needs to be read individually
            for line in tqdm(f):
                data = json.loads(line)
                word = data["word"]
                gloss_list = []
                senses = data["senses"]
                for sense_dict in senses:
                    # If this word is marked as having no gloss, skip it
                    if "tags" in sense_dict.keys() and sense_dict["tags"][0] == "no-gloss":
                        continue
                    # If this word has a cleaned gloss, use this gloss
                    if "glosses" in sense_dict.keys():
                        gloss_key = "glosses"
                    # If there is no cleaned gloss, but the raw gloss is available, we should at least use that
                    elif "raw_glosses" in sense_dict.keys():
                        gloss_key = "raw_glosses"
                    # If neither, skip this entry
                    else:
                        continue
                    gloss = sense_dict[gloss_key]
                    assert len(gloss) == 1
                    if kostiuk_format:
                        gloss = {"definition" : gloss}
                    gloss_list.append(gloss)

                # If there is at least one gloss, then add it to the dictionary, making sure to append
                # In the event of a word having multiple pos
                if gloss_list:
                    dic[word].extend(gloss_list)
    return dic

# Copied from the Kostiuk github, just saves the dictionary to a json in a new directory and encoded as UTF-8
SAVE_DIR = os.path.join(os.pardir, "dictionaries/")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    russian_dict = build_dict(FILES)
    with open(os.path.join(SAVE_DIR, "russian_dictionary.json"), "w", encoding="utf8") as f:
        json.dump(russian_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()