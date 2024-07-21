import json
from tqdm import tqdm
from stop_words import get_stop_words, StopWordError
import networkx as nx
import os
import stanza
from typing import Dict, Any, Set, List


class Dict2Graph:

    """
    Class for converting dictionary into graph
    """

    def __init__(self, stanza_lang: str, stop_words_lang: str, word_dictionary: Dict[str, Any], stanza_dir: str = "",
                 drop_self_cycles: bool = False, lemm_always: bool = True, debug: bool = False) -> None:
        """
        :param stanza_lang: str, lang to use for stanza
        :param stop_words_lang: str, lang to use for stop-words (see stop_words package)
        :param word_dictionary: dict of the following structure:
        {word: [definition_1, definition_2, ...]}
        :param stanza_dir: str, path to dir, where stanza model are stored. Default: ""
        :param drop_self_cycles: bool, if to remove definitions from dict that contain the word they suppose to define
        :param lemm_always: bool: if True, lemmatize all the words in definitions. Otherwise lemmatization will applied
        only if the word is not in the vocabulary
        """

        self.ppl = stanza.Pipeline(
            lang=stanza_lang,
            dir=stanza_dir,
            processors='tokenize,lemma'
        )

        self.word_dictionary = word_dictionary
        self.drop_self_cycles = drop_self_cycles

        self.lemm_always = lemm_always
        self.stop_words_lang = stop_words_lang
        self.debug = debug
        if self.debug:
            self.dropped_words = set()

    def process_defs(self) -> None:
        """
        Bulk process all definitions in the word_dictionary using stanza for later manipulation.
        Saves output over self.word_dictionary, with structure {word: [doc1, doc2, ...]}
        :return: None
        """
        dict_pairs = self.word_dictionary.items()
        num_defs = {k: len(v) for k, v in dict_pairs}
        in_docs = [stanza.Document([], text=d) for word, deflist in dict_pairs for d in deflist]
        out_docs = self.ppl(in_docs)

        i = 0
        for word, _ in dict_pairs:
            self.word_dictionary[word] = []
            num = num_defs[word]
            for j in range(num):
                self.word_dictionary[word].append(out_docs[i + j])
            i += num

    def get_encoding_dict(self) -> Dict[str, int]:
        """
        Building encoding dict for the given vocabulary of self.word_dictionary
        :return: {word: idx}
        """
        return {k.lower():v for v,k in enumerate(self.vocabulary_list)}

    def _get_lemma(self, word) -> str:
        doc = self.ppl(word)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        word_lemma = lemmas[0]
        return word_lemma

    def get_def_tokens(self, definition: stanza.Document) -> Set[str]:
        """
        Retrieve set of tokens from the definition as a stanza Document object. Depending on lemm_always parameter,
        the lemmatization will always be applied or only if the word is not in vocabulary.
        Words that are not in vocabulary will be dropped, and if in debug mode, will be added to dropped_words
        :param definition: stanza.Document, definition to process
        :return: set of str, set of tokens (words)
        """

        if self.lemm_always:
            tokens = [word.lemma for sent in definition.sentences for word in sent.words]
            tokens = [t.lower() for t in tokens if t.lower() in self.word_dictionary]
            tokens = set(tokens)
        else:
            tokens = set()
            for sent in definition.sentences:
                for word in sent:
                    if word.text.lower() in self.word_dictionary:
                        tokens.add(word.text.lower())
                    elif word.lemma.lower() in self.word_dictionary:
                        tokens.add(word.lemma.lower())
                    elif self.debug:
                        self.dropped_words.add(word.text.lower())

        return tokens

    def get_successors(self, word: str) -> Set[str]:
        """
        Gets set of successors from the given word. Self-Cycles will be dropped if self.drop_self_cycles was set to True
        :param word: str, word from vocabulary (self.word_dictionary)
        :return: set of strs (tokens in definitions)
        """

        successors = set()

        for definition in self.word_dictionary[word]:
            try:
                processed_def = self.get_def_tokens(definition=definition)
            except TypeError:
                print(definition, type(definition))
                raise TypeError

            if self.drop_self_cycles:
                if word not in processed_def:
                    successors = successors.union(processed_def)
            else:
                successors = successors.union(processed_def)

        return successors

    def build_graph(self,
                    vocabulary_list: List[str] = None,
                    lemm_vocabulary: bool = False
                    ) -> nx.DiGraph:
        """
        Building a graph of dictionary
        :param vocabulary_list: list of str, list of vocabulary to use. If not provided, vocabulary will be built from keys of word_dictionary.
        :param lemm_vocabulary: bool, ignored if vocabulary_list = None. Whether to lemmatize words in vocabulary_list. The duplicates will be removed
        :return: dict with fields:
            encoding_dict: dict,  encoding dict that was either provided or built
            graph: dict, {word_id: [word_id, word_id, ...]}, graph edges
        """

        # Load stopwords
        try:
            sw = get_stop_words(self.stop_words_lang)
        except StopWordError:
            sw = []

        # Remove stopwords from the dictionary headwords
        self.word_dictionary = {k: v for k, v in self.word_dictionary.items() if k.lower() not in sw}

        if vocabulary_list:
            self.vocabulary_list = vocabulary_list
            if lemm_vocabulary:
                self.vocabulary_list = [self._get_lemma(word) for word in self.vocabulary_list]
        else:
            self.vocabulary_list = list(self.word_dictionary.keys())

        self.process_defs()

        encoding_dict = self.get_encoding_dict()
        digraph = nx.DiGraph()
        # Format the encodings for digraph class' add_nodes_from method
        encoded_nodes = [(v, {"word": k}) for k, v in list(encoding_dict.items())]
        digraph.add_nodes_from(encoded_nodes)

        # Create edges
        encoded_edges = []
        for word in tqdm(self.word_dictionary):
            successors = list(self.get_successors(word=word))
            encoded_edges.extend([(encoding_dict[word], encoding_dict[x]) for x in successors])

        digraph.add_edges_from(encoded_edges)
        return digraph

def build_dict(args):

    with open(args.word_dictionary_path, "r", encoding="utf-8") as f:
        word_dictionary = json.load(f)

    if args.vocabulary_list_path:
        with open(args.vocabulary_list_path, "r", encoding="utf-8") as f:
            vocabulary_list = json.load(f)
    else:
        vocabulary_list = None

    processor = Dict2Graph(
        stanza_dir=args.stanza_dir,
        stanza_lang=args.stanza_lang,
        stop_words_lang=args.stop_words_lang,
        word_dictionary=word_dictionary,
        drop_self_cycles=args.drop_self_cycles,
        lemm_always=args.lemm_always,
        debug=args.debug
    )

    output_dict = processor.build_graph(vocabulary_list=vocabulary_list, lemm_vocabulary=args.lemm_vocabulary)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.debug:
        # os.makedirs(os.path.join("debug"), exist_ok=True)
        with open("debug\\dropped_tokens.txt") as f:
            f.writelines(processor.dropped_words)

    nx.write_graphml(output_dict, os.path.join(args.save_dir, args.stanza_lang + "_graph.graphml"))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Dict to Graph')
    parser.add_argument('--word_dictionary_path', type=str,
                        default="",
                        help='path to word dictionary in json')
    parser.add_argument('--stanza_dir', type=str,
                        default="",
                        help='path to dir, where stanza model are stored')
    parser.add_argument('--stanza_lang', type=str,
                        default="en",
                        help='lang to use for stanza')
    parser.add_argument('--stop_words_lang', type=str,
                        default="english",
                        help='lang to use for stop-words (see stop_words package)')
    parser.add_argument('--save_dir', type=str,
                        default="wordnet_StanzaLemm_NotAlwaysLemmSSC",
                        help='path, where to save results')
    parser.add_argument('--drop_self_cycles', type=bool,
                        default=True,
                        help='drop definitions containing word to define')
    parser.add_argument('--lemm_always', type=bool,
                        default=True,
                        help='Whether to always lemmatize words in definitions or only when it is not found in dict')
    parser.add_argument('--vocabulary_list_path', type=str,
                        default="",
                        help='path to json with vocabulary to use for graph building. If None, the keys from file from word_dictionary_path will be used')
    parser.add_argument('--lemm_vocabulary', type=bool,
                        default=True,
                        help='Ignored if vocabulary_list_path is empty. Whether to lemmatize words in vocabulary_list. The duplicates will be removed')
    parser.add_argument('--debug', type=bool,
                        default=False, help="Debug logging of words dropped because neither the token nor lemma is included in the dictionary")
    args = parser.parse_args()
    build_dict(args)
