import scipy.io
import numpy
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import string

# Helper function to load data
def load_data(prefix, idx_range, jdx_range):
    data_list = []
    for i in idx_range:
        for j in jdx_range:
            file_path = f"/content/drive/MyDrive/data_project/predicted_{prefix}_{i}_{j}.mat"
            data_list.append(scipy.io.loadmat(file_path))
    return data_list

# Helper function to compute accuracy
def compute_accuracy(data_list, label_key):
    count_true, count_total = 0, 0
    for data in data_list:
        prob = data["out_labels"]
        true_labels = data["true_labels"].squeeze()
        pred_labels = numpy.argmax(prob, axis=-1) + 1
        count_total += len(true_labels)
        count_true += numpy.sum(true_labels == pred_labels)
    return count_true, count_total

# Initialize data
all_ssvep = load_data("SSVEP", range(3, 7), range(1, 5))
all_p300 = load_data("P300", range(3, 7), range(1, 5))
target_words = [scipy.io.loadmat(f"/content/drive/MyDrive/data_project/predicted_P300_{i}_{j}.mat")["TestLabels"][0] for i in range(3, 7) for j in range(1, 5)]

# Processing characters
def process_characters(chars):
    chars = [char.lower() for char in chars]
    chars_list = [list(x) for x in chars]
    chars_list[2][2] = "[SPC]"
    return chars_list

Chars = process_characters(["HRT", "IFS", "CN.", "BQV", "OJZ", "XKP", "ALW", "EGD", "MUY"])

# Accuracy computation
count_true, count_total = compute_accuracy(all_ssvep, "SSVEP")
print("accuracy of ssvep without model:", count_true / count_total * 100)

count_true, count_total = compute_accuracy(all_p300, "P300")
print("accuracy of p300 without model:", count_true / count_total * 100)

# Tokenizer and model setup
vocab = list(string.ascii_lowercase) + ["[UNK]"]
label2id = {vocab[i]: i + 1 for i in range(len(vocab))}
id2label = {i + 1: vocab[i] for i in range(len(vocab))}
model = AutoModelForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/models/output/checkpoint-105132",
    problem_type="multi_label_classification",
    num_labels=len(vocab),
    id2label=id2label,
    label2id=label2id
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_word(word):
    tokens = [char if char in vocab else "[UNK]" for char in word.lower()]
    return [label2id[token] for token in tokens]
# Function to simulate an input sequence processing
def split_input_sequence(input):
    y, z = [], []
    for i in range(1, len(input)):
        y.append(input[:i])
        z.append(input[i])
    return y, z

# Function to calculate accuracy with different alpha blending for prediction
def acc_cal_normal(target_words, all_data, alpha, use_softmax, device, model):
    total_count = 0
    total_true = 0
    for i, data in enumerate(all_data):
        prob = data["outlabels"]
        true_labels = data["true_labels"].squeeze()
        ssvep_p = softmax(prob, axis=1) if use_softmax else prob

        input_tokens = torch.tensor(encode_word(target_words[i])).to(device)
        input_seq, true_seq = split_input_sequence(input_tokens)
        for y, z in zip(input_seq, true_seq):
            model_out = model(input_ids=y[None, :])["logits"]
            probs_nlp = torch.nn.functional.softmax(model_out, dim=-1)
            probs_nlp = probs_nlp.cpu().detach().numpy()

            merged_probs = alpha * probs_nlp + (1 - alpha) * ssvep_p
            pred_label = numpy.argmax(merged_probs, axis=-1)

            total_count += 1
            total_true += int(pred_label == z)

    return total_count, total_true

# Function to run experiments over a range of alpha values
def evaluate_alphas(target_words, all_data, device, model, use_softmax):
    range_alpha = numpy.arange(0, 1.05, 0.05)
    best_alpha, best_accuracy = 0, 0
    results = []

    for alpha in range_alpha:
        total_count, total_true = acc_cal_normal(target_words, all_data, alpha, use_softmax, device, model)
        accuracy = total_true / total_count * 100
        results.append(accuracy)
        if accuracy > best_accuracy:
            best_alpha = alpha
            best_accuracy = accuracy

    return range_alpha, results, best_alpha, best_accuracy

# Analysis for SSVEP and P300 using both softmax and no softmax approaches
def perform_analysis():
    for use_softmax in [True, False]:
        print(f"Evaluating {'with' if use_softmax else 'without'} softmax:")
        range_alpha, results, best_alpha, best_accuracy = evaluate_alphas(target_words, all_ssvep, device, model, use_softmax)
        plt.plot(range_alpha, results, label=f"SSVEP {'softmax' if use_softmax else 'no softmax'}")

        range_alpha, results, best_alpha, best_accuracy = evaluate_alphas(target_words, all_p300, device, model, use_softmax)
        plt.plot(range_alpha, results, label=f"P300 {'softmax' if use_softmax else 'no softmax'}")

    plt.xlabel('Alpha')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Alpha Blending')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the analysis
perform_analysis()
