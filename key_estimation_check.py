import json
from random import *

import key_estimation
from main import is_simple_chord

exported_chords_file = "exported-chords.json"

random_numbers = [randint(1, 15000) for _ in range(100)]
cnt = 0
d = {}
output_file = "verify.json"


def satisfied_percentage(iterable, condition):
    satisfied_count = sum(1 for item in iterable if condition(item))
    percentage = (satisfied_count / len(iterable)) * 100 if iterable else 0
    return percentage


def are_notes_same(note1, note2):
    enharmonic_equivalents = {
        'C#': 'Db',
        'D#': 'Eb',
        'F#': 'Gb',
        'G#': 'Ab',
        'A#': 'Bb'
    }
    normalized_note1 = enharmonic_equivalents.get(note1, note1)
    normalized_note2 = enharmonic_equivalents.get(note2, note2)
    return normalized_note1 == normalized_note2


counter = 0
with open("verify.json", 'r') as f:
    estimation = json.load(f)
    for i in range(0, 100):
        if are_notes_same(estimation[str(i)]["guessed_key"], estimation[str(i)]["key"]):
            counter += 1
        else:
            print(estimation[str(i)]["guessed_key"], estimation[str(i)]["key"])
