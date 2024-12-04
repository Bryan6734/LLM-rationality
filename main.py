from minicons import scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
import numpy as np
import torch
import json

prompt = """You are a helpful assistant. I will give you a statement, and you will tell me if it is TRUE or FALSE. Only answer with the word TRUE or FALSE.
Claim: The Eiffel Tower is in London. Answer: FALSE.
Claim: Water boils at 100C at sea level. Answer: TRUE.
Claim: The moon is a star. Answer: FALSE.
Claim: Dogs are mammals. Answer: TRUE.
Claim: Shakespeare wrote “The Great Gatsby.” Answer: FALSE.
Claim: The Amazon River is the longest river in the world. Answer: FALSE.
Claim: Elephants are the largest land animals. Answer: TRUE.
Claim: 2 + 2 = 5. Answer: FALSE.
Claim: Mount Everest is the tallest mountain above sea level. Answer: TRUE.
Claim: A triangle has four sides. Answer: FALSE.
Claim: Tomatoes are technically fruits. Answer: TRUE.
Claim: Lightning is hotter than the surface of the sun. Answer: TRUE.
Claim: The capital of Australia is Sydney. Answer: FALSE.
Claim: The Atlantic Ocean is smaller than the Pacific Ocean. Answer: TRUE.
Claim: Humans have 32 teeth in a complete adult set. Answer: TRUE.C
Claim: $C$ Answer:"""

sentences = [
    "UNC Chapel Hill is the first public university in the United States.",
    "UNC Chapel Hill is not the first public university in the United States.",
    "Aristotle was a philosopher.",
    "Aristotle was not a philosopher.",
    "24 is an integer.",
    "24 is not an integer.",
    "Antarctica is the coldest continent on Earth.",
    "Antarctica is not the coldest continent on Earth.",
    "Joey D. Vieira is an American.",
    "Joey D. Vieira is not an American.",
    "Craig Morton played in the NFL for 18 seasons.",
    "Craig Morton did not play in the NFL for 18 seasons.",
    "Waruhiu Itote was known as General China.",
    "Waruhiu Itote was not known as General China.",
    "Gonzalo Fonseca's work has been exhibited in the Museum of Modern Art.",
    "Gonzalo Fonseca's work has not been exhibited in the Museum of Modern Art.",
    "Paul O'Neill competed in the BTCC for over a decade.",
    "Paul O'Neill did not compete in the BTCC for over a decade.",
    "Chadwick Boseman attended Howard University.",
    "Chadwick Boseman did not attend Howard University."
]

# Create tuples of sentences and their negations
sentence_tuples = [[prompt.replace("$C$", sentences[i]), prompt.replace("$C$", sentences[i + 1])]
                   for i in range(0, len(sentences), 2)]

# Initialize the scorer with GPT-2
model_name = "gpt2"  # or whatever model you're using
score_model = scorer.IncrementalLMScorer(model_name, device="cuda")

results = []
for statement, negation in sentence_tuples:
    # Get token score for "TRUE" for both statements
    statement_true = statement + " TRUE"
    statement_false = statement + " FALSE"

    statement_true_score = score_model.token_score(statement_true)[-1][-1]
    statement_false_score = score_model.token_score(statement_false)[-1][-1]

    negation_true = negation + " TRUE"
    negation_false = negation + " FALSE"
    negation_true_score = score_model.token_score(negation_true)[-1][-1]
    negation_false_score = score_model.token_score(negation_false)[-1][-1]

    # they're tuples for some reason lol
    statement_true_score = list(statement_true_score)
    statement_false_score = list(statement_false_score)
    negation_true_score = list(negation_true_score)
    negation_false_score = list(negation_false_score)

    statement_true_score[-1] = np.exp(statement_true_score[-1])
    statement_false_score[-1] = np.exp(statement_false_score[-1])
    negation_true_score[-1] = np.exp(negation_true_score[-1])
    negation_false_score[-1] = np.exp(negation_false_score[-1])
    
    results.append({
        'statement_true': statement_true,
        'statement_true_score': statement_true_score,
        'statement_false': statement_false,
        'statement_false_score': statement_false_score,
        
        'negation_true': negation_true,
        'negation_true_score': negation_true_score,
        'negation_false': negation_false,
        'negation_false_score': negation_false_score,
    })


# Output the JSON
with open('results_5.json', 'w') as f:
    json.dump(results, f, indent=4)
