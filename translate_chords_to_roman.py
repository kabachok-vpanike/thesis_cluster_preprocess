import re


def note_to_int(note):
    note_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
                'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    return note_map.get(note, None)


def chord_quality(chord):
    if "aug7" in chord or "7#5" in chord:
        return "aug7"
    elif "sus2" in chord:
        return "sus2"
    elif "sus4" in chord:
        return "sus4"
    elif "m7b5" in chord:
        return "half-dim"
    elif "dim7" in chord:
        return "dim7"
    elif "maj7" in chord:
        return "maj7"
    elif "m7" in chord:
        return "m7"
    elif "dim" in chord:
        return "dim"
    elif "aug" in chord:
        return "aug"
    elif "7" in chord:
        return "7"
    elif "m" in chord:
        return "m"
    else:
        return "major"


def scale_degrees_to_roman(degree_index, chord_quality, key_context):
    base_roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][degree_index]

    if key_context == 'minor':
        base_roman = base_roman.lower()

    if chord_quality == "dim7":
        return base_roman + "°7"
    elif chord_quality == "half-dim":
        return base_roman + "ø7"
    elif chord_quality == "aug7":
        return base_roman.upper() + "+7"
    elif chord_quality in ["sus2", "sus4"]:
        return base_roman + chord_quality
    elif chord_quality == "m7":
        return base_roman.lower() + "7"
    elif chord_quality == "7":
        return base_roman + "7"
    elif chord_quality == "dim":
        return base_roman.lower() + "°"
    elif chord_quality == "aug":
        return base_roman + "+"
    elif chord_quality == "m":
        return base_roman.lower()
    else:
        return base_roman


def get_chord_root(chord):
    chord = chord.split('/')[0]
    root_note_pattern = re.compile(r"^[A-G](b{1,2}|#{1,2})?")
    match = root_note_pattern.match(chord)

    if match:
        return match.group()
    else:
        return "Invalid chord notation"


def transpose_chord_to_key(chord, key):
    major_scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    harmonic_minor_scale_intervals = [0, 2, 3, 5, 7, 8, 11]

    key_root = note_to_int(key[:-1] if key.endswith('m') else key)
    chord_root = note_to_int(get_chord_root(chord))
    chord_qual = chord_quality(chord)

    try:
        interval = (chord_root - key_root) % 12
    except:
        return 'wrong symbol'
    key_scale = harmonic_minor_scale_intervals if key.endswith('m') else major_scale_intervals

    if interval in key_scale:
        degree_index = key_scale.index(interval)
    else:
        return "Chord out of scale"
    key_context = 'minor' if key.endswith('m') else 'major'
    roman_numeral = scale_degrees_to_roman(degree_index, chord_qual, key_context)

    return roman_numeral


def translate_chords_to_roman(chords, key):
    res = []
    for chord in chords:
        translated = transpose_chord_to_key(chord, key)
        res.append(translated if translated != 'wrong symbol' else 'X')
    return res
