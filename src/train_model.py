import argparse
from data_processing import load_and_prepare_data
from custom_dataset import TextDataset
from model_config import create_model
from training import setup_and_train
from visualization import save_plot

def parse_args():
    parser = argparse.ArgumentParser(description="Run the text processing pipeline.")
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model to be used')
    parser.add_argument('--output_dir', type=str, default='models/output', help='Directory for output and logs')
    parser.add_argument('--plot_file', type=str, default='training_metrics.png', help='Filename for saving the training plot')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset, label2id, id2label = load_and_prepare_data()
    vocab_size = len(label2id)

    train_data = TextDataset(dataset['train']['tokens'], dataset['train']['labels'])
    val_data = TextDataset(dataset['validation']['tokens'], dataset['validation']['labels'])

    model = create_model(vocab_size, id2label, label2id, model_name=args.model_name)
    trainer = setup_and_train(model, train_data, val_data, args.output_dir)

    save_plot(trainer.state.log_history, args.plot_file)

if __name__ == "__main__":
    main()
