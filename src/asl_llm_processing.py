from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model setup - using a pre-trained text-to-text model (FLAN-T5) to try to turn ASL letters into words
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)  # Loads tokenizer for converting text to model input
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Loads the actual model

# These are the letters predicted by our ASL recognition models from video frames
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


# HELPER FUNCTIONS
def collapse_repeats(text, min_repeats=3):
    """
    Combine repeated letters into one if they repeat enough times.
    Example: 'AAAABBB' -> 'AB' (if min_repeats=3)
    This helps remove the extra repeated letters from ASL predictions.
    """
    if not text:
        return ""

    cleaned = []  # To store collapsed letters
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1  # Same letter repeated
        else:
            if count >= min_repeats:
                cleaned.append(text[i - 1])  # Add the letter once
            count = 1

    if count >= min_repeats:  # Handle last letters
        cleaned.append(text[-1])

    return "".join(cleaned)


def llm_reconstruct(sequence):
    """
    Send the collapsed letters to the LLM and ask it to form words/sentences.
    """
    if not sequence:
        return ""

    # Create a prompt for the model
    prompt = (
        "The following letters are from ASL fingerspelling. "
        "The letters are in order, do not mix them up. "
        "Some sequences are words, others are short sentences. "
        "Convert this sequence into proper English words or sentences: "
        f"{sequence}"
    )

    # Convert the prompt into tokens for the model
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the model output
    output = model.generate(**inputs, max_new_tokens=64)

    # Convert tokens back to readable text
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# MAIN - Loop through all ASL predictions and try to reconstruct words/sentences
for file, raw_letters in predictions.items():
    print(f"=== Processing {file} ===")
    print(f"Raw letters: {raw_letters}")

    # Collapse repeated letters first
    collapsed = collapse_repeats(raw_letters, min_repeats=3)
    print(f"Collapsed letters: {collapsed}")

    # Use the LLM to reconstruct words
    reconstructed = llm_reconstruct(collapsed)
    print(f"LLM final output: {reconstructed}\n")
