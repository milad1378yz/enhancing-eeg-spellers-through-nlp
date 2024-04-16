import argparse
from data_loader import load_brain_signals
from analysis import compute_ssvep_accuracy, compute_p300_accuracy
from model_evaluation import initialize_model, tokenize_text
from visualization import save_accuracy_plot

def main():
    parser = argparse.ArgumentParser(description="Process and analyze brain signal data.")
    parser.add_argument('--data_path', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--model_path', type=str, default='models/output/checkpoint-105132', help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device (cuda or cpu)')
    parser.add_argument('--plot_filename', type=str, default='accuracy_plot.png', help='Filename for the saved accuracy plot')
    args = parser.parse_args()

    # Load data
    p300_signals, ssvep_signals, _ = load_brain_signals(args.data_path, range(3, 7), range(1, 5))

    # Compute accuracies
    ssvep_accuracy = compute_ssvep_accuracy(ssvep_signals)
    p300_accuracy = compute_p300_accuracy(p300_signals)
    print(f"SSVEP Accuracy: {ssvep_accuracy[0]}%, P300 Accuracy: {p300_accuracy[0]}%")

    # Initialize model and perform evaluations
    model, label2id, _ = initialize_model(args.model_path, args.device)

    # Example usage of tokenizer
    # example_tokens = tokenize_text("example", label2id)

    # Optionally, save a plot
    alpha_range = [0.1 * i for i in range(10)]
    accuracy_scores = [ssvep_accuracy[0] for _ in alpha_range] 
    save_accuracy_plot(alpha_range, accuracy_scores, "SSVEP Model Accuracy Over Alpha", args.plot_filename)

if __name__ == '__main__':
    main()
