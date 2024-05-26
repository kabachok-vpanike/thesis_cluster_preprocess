from translate_chords_to_roman import translate_chords_to_roman


def translate_chords_to_roman_check():
    test_cases = [
        (("C", "C"), ["I"]),
        (("G", "C"), ["V"]),
        (("Am", "C"), ["vi"]),
        (("Em", "G"), ["vi"]),
        (("Bb", "F"), ["IV"]),
        (("D#m", "B"), ["iii"]),
        (("F#", "D"), ["III"]),
        (("C#m", "A"), ["iii"]),
        (("Gb", "Db"), ["IV"]),
        (("B", "E"), ["V"]),
        (("Ebm", "Bb"), ["iv"]),
        (("Ab", "Eb"), ["IV"]),
        (("Cm", "Eb"), ["vi"]),
        (("F", "C"), ["IV"]),
        (("A", "D"), ["V"]),
        (("E", "A"), ["V"]),
        (("Bbm", "F"), ["iv"]),
        (("Dm", "G"), ["v"]),
        (("G#m", "E"), ["iii"]),
        (("F#m", "A"), ["vi"]),
        (("Bdim", "C"), ["vii°"]),
        (("Edim", "Am"), ["v°"]),
        (("Gaug", "C"), ["V+"]),
        (("Daug", "G"), ["V+"]),
        (("A7", "D"), ["V7"]),
        (("E7", "A"), ["V7"]),
        (("Bbm7", "F"), ["iv7"]),
        (("Cm7", "Fm"), ["v7"]),
        (("Dsus2", "G"), ["Vsus2"]),
        (("Asus4", "D"), ["Vsus4"]),
        (("F#m7b5", "Bm"), ["vø7"]),
        (("C#aug7", "F#m"), ["V+7"]),
        (("Gdim7", "Cm"), ["v°7"]),
        # (("Ebm7", "Ab"), ["IVM7"]),
        (("Bsus2", "E"), ["Vsus2"]),
        (("G#m7", "C#m"), ["v7"]),
        (("D#dim", "B"), ["iii°"]),
        (("A#aug", "D#m"), ["VI+"]),
        (("Csus4", "F"), ["Vsus4"]),
        (("Fm7", "Bbm"), ["v7"]),
    ]

    passed_tests = 0

    for i, (inputs, expected_output) in enumerate(test_cases, start=1):
        chord, key = inputs
        result = translate_chords_to_roman([chord], key)
        if result == expected_output:
            print(f"Test Case #{i}: PASSED")
            passed_tests += 1
        else:
            print(f"Test Case #{i}: FAILED - Expected {expected_output}, got {result}")
    total_tests = len(test_cases)
    print(f"\n{passed_tests}/{total_tests} tests passed.")


translate_chords_to_roman_check()
