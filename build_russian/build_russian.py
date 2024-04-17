import json
import os
from tqdm import tqdm

russian_dict = {}

# NOTE: Only use the words in Russian tagged as having pos adj, adv, noun, or verb, since these are considered "lexical"
# See the Primes one for more details
files = ["kaikki.org-dictionary-Русский-by-pos-adj.json", "kaikki.org-dictionary-Русский-by-pos-adv.json",
             "kaikki.org-dictionary-Русский-by-pos-noun.json", "kaikki.org-dictionary-Русский-by-pos-verb.json"]

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
                # If this word has a cleaned gloss, add it
                if "glosses" in sense_dict.keys():
                    assert len(sense_dict["glosses"]) == 1
                    gloss_list.append({"definition": sense_dict["glosses"][0]})
                # If the gloss cannot be cleaned, and only the raw gloss is available, at least use that
                elif "raw_glosses" in sense_dict.keys():
                    assert len(sense_dict["raw_glosses"]) == 1
                    gloss_list.append({"definition": sense_dict["raw_glosses"][0]})

            # If there is at least one gloss, then add it to the dictionary, making sure to append
            # In the event of a word having multiple pos
            if gloss_list:
                try:
                    russian_dict[word].extend(gloss_list)
                except KeyError:
                    russian_dict[word] = gloss_list

# Copied from the github, just saves the dictionary to a json in a new directory and encoded as UTF-8
SAVE_DIR = os.path.join(os.pardir, "dictionaries/")
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, "russian_dictionary.json"), "w", encoding="utf8") as f:
   json.dump(russian_dict, f, ensure_ascii=False)
