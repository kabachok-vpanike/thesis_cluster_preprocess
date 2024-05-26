from typing import List
import numpy as np
import re
from translate_chords_to_roman import get_chord_root


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    total = sum(vector)
    if total > 0:
        return vector / total
    return vector


def generate_pitch_class_distribution(chords: List[List[int]]) -> np.ndarray:
    distribution = np.zeros(12)
    for chord in chords:
        for note in chord:
            distribution[note % 12] += 1
    return normalize_vector(distribution)


def calculate_correlations(distribution: np.ndarray, profile: np.ndarray) -> List[float]:
    return [float(np.corrcoef(np.roll(profile, i), distribution)[0, 1]) for i in range(12)]


def find_best_match(major_correlations: List[float], minor_correlations: List[float]) -> str:
    max_major_idx, max_major_val = max(enumerate(major_correlations), key=lambda x: x[1])
    max_minor_idx, max_minor_val = max(enumerate(minor_correlations), key=lambda x: x[1])

    if max_major_val > max_minor_val:
        return f"{note_name(max_major_idx)}"
    else:
        return f"{note_name(max_minor_idx)}m"


def note_name(note_number: int) -> str:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return notes[note_number % 12]


# Note to pitch class mapping
note_to_pitch_class = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}


def extract_root(chord):
    chord = chord.split('/')[0]
    match = re.match(r'^[A-Ga-g][#b]?', chord)
    if match:
        return match.group()
    return None


def chord_name_to_pitch_classes(chord_name: str) -> List[int]:
    is_minor = chord_name.endswith('m')
    root_note = get_chord_root(chord_name)
    root_pitch_class = note_to_pitch_class[root_note]
    if is_minor:
        return [root_pitch_class, (root_pitch_class + 3) % 12, (root_pitch_class + 7) % 12]
    else:
        return [root_pitch_class, (root_pitch_class + 4) % 12, (root_pitch_class + 7) % 12]


def chords_to_pitch_classes(chords: List[str]) -> List[List[int]]:
    return [chord_name_to_pitch_classes(chord_name) for chord_name in chords]


def estimate_key(chords: List[str]) -> str:
    chords_as_pitch_classes = chords_to_pitch_classes(chords)
    major_profile = normalize_vector(np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]))
    minor_profile = normalize_vector(np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]))

    distribution = generate_pitch_class_distribution(chords_as_pitch_classes)
    major_correlations = calculate_correlations(distribution, major_profile)
    minor_correlations = calculate_correlations(distribution, minor_profile)

    return find_best_match(major_correlations, minor_correlations)
