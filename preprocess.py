import re

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[.!?]", re.UNICODE)
SENTENCE_RE = re.compile(r'[^.!?]+[.!?]?')


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def split_sentences(text):
    for match in SENTENCE_RE.finditer(text):
        raw = match.group().strip()
        tokens = tokenize(raw)
        ending = tokens[-1] if tokens and tokens[-1] in '.!?' else '.'
        words = [token for token in tokens if token not in '.!?']
        if words:
            yield words, ending


def format_sentence(words, ending='.'):
    if not words:
        return ''
    text = ' '.join(words)
    return text[0].upper() + text[1:] + ending
