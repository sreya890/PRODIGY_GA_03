from src.markovmodel import MarkovModel

# -----------------------------------------
# CREATE MODEL
# -----------------------------------------

model = MarkovModel()


# -----------------------------------------
# READ TRAINING DATA
# -----------------------------------------

with open("data.txt", "r", encoding="utf-8") as file:

    text = file.read()


# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------

model.train(text)


# -----------------------------------------
# CALCULATE PROBABILITIES
# -----------------------------------------

model.calculate_probabilities()


# -----------------------------------------
# DISPLAY PROBABILITIES
# -----------------------------------------

print("\nMarkov Chain Probabilities")
print("==========================")

model.display_probabilities()


# -----------------------------------------
# TEST WEIGHTED RANDOM SELECTION
# -----------------------------------------

print("\n\nTesting weighted random selection:")
print("----------------------------------")

current_word = "intelligence"

for i in range(10):

    next_word = model.generate_next_word(
        current_word
    )

    print(next_word)


# -----------------------------------------
# GENERATE TEXT
# -----------------------------------------

print("\n\nGenerated Text:")
print("----------------")

generated_text = model.generate_text(
    "artificial",
    20
)

print(generated_text)