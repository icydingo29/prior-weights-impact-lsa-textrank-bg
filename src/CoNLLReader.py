import pathlib
import numpy as np
import re
from io import open
from collections import defaultdict
from bulstem.stem import BulStemmer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent # change if this file is moved

# Bulstem constants
RULES_DIR = BASE_DIR / "src" / "bulstem" / "stemrules"
PRE_DEFINED_RULES = [
    RULES_DIR / 'stem_rules_context_1_utf8.txt',
    RULES_DIR / 'stem_rules_context_2_utf8.txt',
    RULES_DIR / 'stem_rules_context_3_utf8.txt'
]
STOPWORDS_FILE_PATH  = BASE_DIR / "resources" / "BTB-StopWordList.txt"

# Constants for pre-defined names of parts-of-speech sets
STRATEGY_MINIMAL = {'NOUN', 'PROPN'}
STRATEGY_DESCRIPTIVE = {'NOUN', 'PROPN', 'ADJ'}
STRATEGY_FULL = {'NOUN', 'PROPN', 'ADJ', 'VERB'} # Default

# Constants for sentence priors 
ALPHA = 0 # pos-score preference
BETA = 1 - ALPHA #ner score preference

# Constants for pos tags and ann tags weights
POS_TAGS_WEIGHTS = {'NOUN': 1.0, 'PROPN': 1.0, 'ADJ': 0.6, 'VERB': 0.4}
NER_TAGS_WEIGHTS = {
    # Persons
    'B-PER': 2.0, 'I-PER': 2.0,
    
    # Organisations
    'B-ORG': 1.5, 'I-ORG': 1.5, 
    'B-ORG:B-ORG': 1.5, # there are such tags in the dataset
    
    # Locations
    'B-LOC': 1.5, 'I-LOC': 1.5,
    'B-LOC:B-LOC': 1.5, 'I-LOC:I-LOC': 1.5,
    
    # Events
    'B-EVT': 1.2, 'I-EVT': 1.2,
    
    # Products
    'B-PRO': 1.0, 'I-PRO': 1.0,
    
    # Everything else
    'O': 0.0 
}

# Contants for reading conll file
TOKEN_ID_IDX = 1
RAW_WORD_IDX = 2
POS_TAG_IDX = 3
ANN_TAG_IDX = 6

# stemmer singleton
_stemmer = BulStemmer.from_file(str(RULES_DIR / 'stem_rules_context_2_utf8.txt'), min_freq=2, left_context=2)

signs = {"," : 1, "." : 1, "!" : 1, "?" : 1, "“" : 1, # Space after, but not before
         "-" : 2,                                     # Space before and after
         "„" : 3}                                     # Space before and not after

def normalizeSpaceAroundSigns(original_sentence):
    s = re.sub(r"\s+", " ", " ".join(original_sentence)).strip()
    for ch, rule in signs.items():
        esc = re.escape(ch)
        s = re.sub(rf"\s*{esc}\s*", {1:f"{ch} ", 2:f" {ch} ", 3:f" {ch}"}[rule], s)
    return re.sub(r"\s+", " ", s).strip()

def load_stopwords_set(file_path=STOPWORDS_FILE_PATH):
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        stopwords = set()
        for line in lines:
            stopwords.add(line.rstrip('\n'))
        return stopwords
    except FileNotFoundError:
        print(f"Error, file {file_path} not found!")
        exit()
    
# stopwords singleton   
_stopwords = load_stopwords_set()

def process_conll_file(file_path, strategy_set = STRATEGY_FULL):
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()

        original_text_lines, edited_text_lines = list(), list()
        lines_pos_tags_encounters, lines_ann_tags_encounters = list(), list()

        original_sentence = list()
        edited_sentence = defaultdict(int)
        sent_pos_tags_encounters = defaultdict(int)
        sent_ann_tags_encounters = defaultdict(set)

        def finalize_sentence():
            """Helper function to add sentence data to lists."""
            if not original_sentence: return
            current_s = normalizeSpaceAroundSigns(original_sentence) #" ".join(original_sentence).strip()
    
            # if this sentence is the same as the previous one, ignore it
            if original_text_lines and current_s == original_text_lines[-1]:
                    return 

            original_text_lines.append(current_s)
            edited_text_lines.append(dict(edited_sentence)) # Hard copy 
            lines_pos_tags_encounters.append(dict(sent_pos_tags_encounters))
            lines_ann_tags_encounters.append({k: set(v) for k, v in sent_ann_tags_encounters.items()})

        for line in lines[1:]:
            line = line.strip()
            if not line:
                finalize_sentence()
                original_sentence, edited_sentence = [], defaultdict(int)
                sent_pos_tags_encounters = defaultdict(int)
                sent_ann_tags_encounters = defaultdict(set)
                continue

            tokens = line.split('\t')
            if len(tokens) <= ANN_TAG_IDX: continue

            word = tokens[RAW_WORD_IDX]
            original_sentence.append(word)
            
            lowered_word = word.lower()
            if lowered_word in _stopwords: continue 
            
            ann_tag = tokens[ANN_TAG_IDX]
            stemmed_word = _stemmer.stem(lowered_word)

            if ann_tag != 'O':
                sent_ann_tags_encounters[ann_tag].add(stemmed_word)

            if (tokens[POS_TAG_IDX] in strategy_set):
                edited_sentence[stemmed_word] += 1
                sent_pos_tags_encounters[tokens[POS_TAG_IDX]] += 1
     
        finalize_sentence()

        return original_text_lines, edited_text_lines, lines_pos_tags_encounters, lines_ann_tags_encounters

    except FileNotFoundError:
        print(f"Error, file {file_path} not found!")
        exit()

def calculate_sentence_priors(original_sentences, pos_dicts, ann_dicts):
    n = len(original_sentences)
    pos_scores = np.zeros(n)
    ner_scores = np.zeros(n)

    for i in range(n):
        words = original_sentences[i].split()
        length = len(words)
        if length == 0: continue

        # POS Score
        pos_scores[i] = sum(POS_TAGS_WEIGHTS.get(t, 0) * pos_dicts[i].get(t, 0) 
                            for t in POS_TAGS_WEIGHTS.keys()) / length
        
        # NER Score
        ner_scores[i] = sum(NER_TAGS_WEIGHTS.get(t, 0) * len(ann_dicts[i].get(t, set())) 
                            for t in NER_TAGS_WEIGHTS.keys()) / length

    priors = (ALPHA * pos_scores) + (BETA * ner_scores)
    
    # Normalization
    total_sum = np.sum(priors)
    return priors / total_sum if total_sum > 0 else np.ones(n) / n

if __name__ == "__main__":
    pass


