from src.markovmodel import MarkovModel

text = ('As a result, predictive analytics supports planning. '
        'As a result, machine learning improves predictions. '
        'Machine learning supports better decisions! '
        'Can machine learning help?')
model = MarkovModel(order=3)
model.train(text)
assert model.is_sentence_start('as a result')
assert not model.is_sentence_start('a result predictive')
assert model.get_probabilities('as a result') == {'predictive': 0.5, 'machine': 0.5}
assert model.generate_text('as a result', 8).startswith('As a result')
assert 'a result predictive' not in model.available_starts()
print('All checks passed.')
print(model.generate_text('as a result', 20))
