import random
from collections import Counter, defaultdict

from .preprocess import tokenize, split_sentences, format_sentence


class MarkovModel:
    """Sentence-aware word Markov chain with optional repetition control."""

    def __init__(self, order=2, temperature=1.0):
        if order not in (1, 2, 3):
            raise ValueError('Order must be 1, 2, or 3')
        if temperature <= 0:
            raise ValueError('Temperature must be positive')
        self.order = order
        self.temperature = temperature
        self.model = {}
        self.words = []
        self.start_counts = Counter()
        self.training_sentences = set()

    def preprocess(self, text):
        return tokenize(text)

    def train(self, text):
        counts = defaultdict(Counter)
        self.words = []
        self.start_counts = Counter()
        self.training_sentences = set()
        for words, ending in split_sentences(text):
            self.words.extend(words)
            self.start_counts[tuple(words[:self.order])] += 1
            self.training_sentences.add(' '.join(words))
            sequence = ['<START>'] * self.order + words + [ending]
            for i in range(len(sequence) - self.order):
                counts[tuple(sequence[i:i + self.order])][sequence[i + self.order]] += 1
        self.model = dict(counts)

    def get_probabilities(self, state):
        if isinstance(state, str):
            state = tuple(self.preprocess(state))
        counts = self.model.get(tuple(state), {})
        total = sum(counts.values())
        return {word: count / total for word, count in counts.items()} if total else {}

    def generate_next_word(self, state, temperature=None):
        if isinstance(state, str):
            state = tuple(self.preprocess(state))
        counts = self.model.get(tuple(state))
        if not counts:
            return None
        temp = self.temperature if temperature is None else temperature
        if temp <= 0:
            raise ValueError('Temperature must be positive')
        choices = list(counts)
        weights = [count ** (1 / temp) for count in counts.values()]
        return random.choices(choices, weights=weights, k=1)[0]

    def available_starts(self, limit=20):
        return [' '.join(state) for state, _ in self.start_counts.most_common(limit)]

    def is_sentence_start(self, start_text):
        state = tuple(self.preprocess(start_text))
        return state in self.start_counts

    def _generate_sentence(self, state, max_new_tokens):
        state = tuple(state)
        output = [word for word in state if word != '<START>']
        for _ in range(max_new_tokens):
            word = self.generate_next_word(state)
            if word is None:
                return output, '…', False
            if word in ('.', '!', '?'):
                return output, word, True
            output.append(word)
            state = tuple((list(state) + [word])[-self.order:])
        return output, '…', False

    def generate_text(self, start_text='', length=50, max_extra=40,
                      avoid_repetition=True, retries=8, strict_start=True):
        if not self.model:
            return 'Please train the model first.'
        if length < 1 or max_extra < 0:
            return 'Length must be positive and max_extra cannot be negative.'
        start = [word for word in self.preprocess(start_text) if word not in '.!?']
        if start:
            if len(start) != self.order:
                return f'Enter exactly {self.order} starting words, or leave the field empty.'
            initial = tuple(start)
            if initial not in self.model:
                return 'Starting sequence not found in the training data.'
            if strict_start and initial not in self.start_counts:
                return ('This sequence occurs inside sentences, not at the beginning. '
                        'Choose a suggested sentence start or disable strict_start.')
        else:
            initial = ('<START>',) * self.order

        sentences, seen = [], set()
        used = 0
        while used < length:
            state = initial if not sentences else ('<START>',) * self.order
            remaining = length - used
            # The initial context counts toward the requested word total.
            context_words = sum(word != '<START>' for word in state)
            budget = max(1, remaining - context_words) + max_extra
            chosen = None
            for _ in range(max(1, retries if avoid_repetition else 1)):
                words, ending, complete = self._generate_sentence(state, budget)
                if not words:
                    continue
                signature = ' '.join(words)
                if not avoid_repetition or signature not in seen:
                    chosen = (words, ending, complete, signature)
                    break
            if chosen is None:
                break
            words, ending, complete, signature = chosen
            seen.add(signature)
            sentences.append(format_sentence(words, ending))
            used += len(words)
            if not complete:
                break
        return ' '.join(sentences)

    def get_statistics(self):
        return {
            'total_tokens': len(self.words),
            'unique_tokens': len(set(self.words)),
            'states': len(self.model),
            'order': self.order,
            'sentence_starts': len(self.start_counts),
        }
