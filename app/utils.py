# app/utils.py
import yake
import spacy

_kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
_nlp = spacy.load("en_core_web_sm")

def extract_keywords(text, max_keywords=20):
    keys = _kw_extractor.extract_keywords(text)
    return [k for k, score in keys][:max_keywords]

def extract_entities(text):
    doc = _nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
