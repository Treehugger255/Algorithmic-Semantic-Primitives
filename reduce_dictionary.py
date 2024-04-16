import os
import json
import random

#
with open("dictionaries\\russian_dictionary.json", "r", encoding="utf-8") as f:
    dic: dict = json.load(f)

random.seed(42)

keys = list(dic.keys())
reduced_keys = random.sample(keys, 1000)

reduced_dic: dict = {key : dic[key] for key in reduced_keys}
assert len(reduced_dic) == 1000

with open("russian_reduced.json", "w", encoding="utf-8") as f:
    json.dump(reduced_dic, f, ensure_ascii=False)