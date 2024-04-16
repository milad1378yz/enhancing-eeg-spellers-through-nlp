from datasets import load_dataset
import string

def load_and_prepare_data():
    vocab = list(string.ascii_lowercase) + ["[UNK]"]
    label2id = {vocab[i]: i + 1 for i in range(len(vocab))}
    id2label = {i + 1: vocab[i] for i in range(len(vocab))}

    dataset = load_dataset("glue", "cola")

    def process_sentence(sentence):
        return "".join(sentence.replace(",", "").replace("'", "").split()).lower()

    def custom_tokenizer(sentence):
        tokens = []
        for char in sentence:
            if char.isalpha() and (char in label2id):
                tokens.append(label2id[char])
            else:
                tokens.append(label2id["[UNK]"])  # Use [UNK] token for unknown characters
        return tokens

    for split in dataset:
        dataset[split] = dataset[split].map(
            lambda example: {
                "sentence": process_sentence(example["sentence"]),
                "tokens": custom_tokenizer(process_sentence(example["sentence"]))
            }
        )

    return dataset, label2id, id2label
