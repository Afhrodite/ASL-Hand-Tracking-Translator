from spellchecker import SpellChecker

# These are letters predicted from videos for each sample
predictions = {
    "beach.npy": "BBBBBBBBEEEEEEEEAAAAAAAAADCCDCCCCCCCCCCCHHHHHHHHH",
    "cat.npy": "CCCCCCCCCCCCCCCCCCCCCCCCCCCNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "face.npy": "FFFFFFFFFFFAAAAADDDCCCCEEEEEEEE",
    "face_of_a_gem.npy": "FFFFFFFFAAAAAACCCCCCCCBBEEEEEEEEDDDDDDDDDEEDDDDDDFFFFFFFFFFFFFFAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGBEEEEEEEEEFRMMMMMMMMMMA",
    "he_has_a_face.npy": "HHHHHHHHHHEEEEEEEEEEDHHHHHHHHHCAAAAAAAAADDDDSSSSSSSSSDAAAAAAAAAFFFFFFFAAAAAACCCCCCCCCCEEEEEEEBEE",
    "he_is_mad.npy": "HHHHHHHHHAEEEEEEEEEEEIIIIIIIIIIIIIIIDSSSSSSSSSSSSSSMMAMMMMMMMMMFAAAAAAAAAAACCDDDCCDDDDDDDD",
    "ice.npy": "IIIIIIIIICCCCCCCCCCEEEEEEEE",
    "image.npy": "IIIIIIIRRMMMNMMMMTAAAAAAAAAOGGGGGGGGGEEEEPEEEEEEE",
    "i_am.npy": "IIIIIIIIIIIIIIIIAAAAAAAAAADRMMMMMMMMMMM",
    "i_need_a_bag.npy": "IIIIIIIIIIIIIIIAANNAAAAAAAAAEEEEEEEEEEEEEEEEDDDDDDDDDDDDDAAAAAAAAAAAAAABBBBBBBBBAAAAAAAAAAKGGGGGGGGGGG",
}

# Initialize a spell checker
spell = SpellChecker()

# Short valid words that are okay even if they are just 1-2 letters
valid_small_words = {
    "i",
    "a",
    "he",
    "am",
    "we",
    "be",
    "go",
    "to",
    "so",
    "in",
    "on",
    "at",
    "me",
    "it",
    "is",
}


# Collapse repeated letters
def collapse_repeats(text, min_repeats=3):
    """
    Combine repeated letters if they appear >= min_repeats.
    Example: 'AAAABBB' -> 'AB'
    This cleans up ASL predictions that repeat letters a lot.
    """
    if not text:
        return ""

    out, count = [], 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1  # keep counting repeated letters
        else:
            if count >= min_repeats:
                out.append(text[i - 1])  # add letter once if repeated enough
            count = 1

    if count >= min_repeats:
        out.append(text[-1])  # handle last letters

    return "".join(out)


# Check if a string is a real word
def is_word(chunk):
    """
    Returns True if chunk is a real word or in the allowed short words list.
    """
    chunk = chunk.lower()

    if chunk in valid_small_words:
        return True  # allow known short words

    if len(chunk) <= 2:
        return False  # ignore small unknown chunks

    return chunk in spell  # check with spellchecker


# Greedy splitting of collapsed letters
def greedy_split(text):
    """
    Split a string of letters into words by checking longest possible valid words first.
    If no word is found, keep single letters.
    """
    text = text.lower()

    words = []
    i = 0
    while i < len(text):
        found = False
        # check from longest possible slice down to shortest
        for j in range(len(text), i, -1):
            piece = text[i:j]
            if is_word(piece):
                words.append(piece)
                i = j  # move index past the found word
                found = True
                break
        if not found:
            words.append(text[i])  # keep single letter if no word found
            i += 1
    return words


# MAIN - Process all predictions
for file, raw in predictions.items():
    print(f"\n=== {file} ===")

    # Collapse repeated letters first
    collapsed = collapse_repeats(raw)
    print("Collapsed:", collapsed)

    # Split collapsed string into words
    result = greedy_split(collapsed)

    # Capitalize first word for readability
    result[0] = result[0].capitalize()

    print("Prediction:", " ".join(result))
