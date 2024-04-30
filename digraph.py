import argparse
import json
import os
import rustworkx as rx
import stanza
import stopwords
from tqdm import tqdm


def format_dictionary(dictionary: dict, lang: str, logging_level: str) -> None:
    """Tokenize and lemmatize dictionary and remove stopwords."""
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma", logging_level=logging_level)
    sw = stopwords.get_stopwords(lang)

    for word in tqdm(dictionary, desc="Tokenizing, lemmatizing, and removing stopwords from dictionary"):
        definitions = dictionary[word]
        for i in range(len(definitions)):
            defn_info = nlp(definitions[i])
            definitions[i] = [word.lemma for sent in defn_info.sentences for word in sent.words if word.lemma not in sw]


def make_digraph(dictionary: dict, check_cycle=False) -> rx.PyDiGraph:
    """
    Take in a formatted dictionary and turn it into a digraph.

    :param dictionary: dictionary with each key as a words and each corresponding values as list of definitions of that word (tokenized and lemmatized).
    :type dictionary: dict
    :param check_cycle: Checks if the addition of an edge creates a cycle during the digraph creation.
    :type check_cycle: bool
    :rtype: rx.PyDigraph
    """
    check_cycle = True # IMPORTANT: If set to True, throws a DAGWouldCycle error whenever adding an edge would create a cycle (and does not add the edge). Cycles created otherwise.
    digraph = rx.PyDiGraph(check_cycle=check_cycle, multigraph=False) # I think we shouldn't allow for multigraphs... I don't see why a word should point to a word in its definition more than once

    # Question: If a word doesn't have a definition: wouldn't it be a good idea to consider it a semantic prime?

    # Step 1: Every entry of the dictionary is a node
    digraph.add_nodes_from(list(dictionary.keys())) 

    # Step 2: Add edges between word and each word in its definitions only if the words in their definitions are already nodes (i.e. they were already in the dictionary)
    nodes = digraph.nodes()
    for node in tqdm(nodes, desc="Creating digraph"):
        definitions = dictionary[node]
        # ugly nested loop, wish there was a way around this
        for definition in definitions:
            for word in definition:
                if word in nodes:
                    try:
                        digraph.add_edges_from([(nodes.index(node), nodes.index(word), tuple)])
                    except rx.DAGWouldCycle:
                        ... # TODO: possibly add to list of possible semantic primes?
    return digraph


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="digraph.py",
        description="Converts .json word dictionaries into digraphs representing when words refer to each other, where dictionary entry is in the form word : list of definitions."
    )
    parser.add_argument("dictionary",
                        help="Path to .json dictionary.")
    parser.add_argument("-s", "--save_path", default="graphs/", 
                        help="Directory for where the graph data is saved.")
    parser.add_argument("-l", "--lang", default="en", 
                        help="Stanza and stopwords language.")
    parser.add_argument("--logging_level", default='INFO',
                        help="Logging level for stanza pipeline.")
    parser.add_argument("--print_visualization", default=False,
                        help="Prints out matplotlib representation of digraph.")

    args = parser.parse_args()

    with open(args.dictionary, "r", encoding="utf-8") as f:
        dictionary = json.load(f)
        format_dictionary(dictionary, lang=args.lang, logging_level=args.logging_level)

    print("Making digraph...")
    rx.node_link_json(make_digraph(dictionary), path=args.save_path)
    print("Done!")


if __name__ == "__main__":
    main()