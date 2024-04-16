from transformers import AutoModelForSequenceClassification

def create_model(vocab_size, id2label, label2id, model_name='gpt2'):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=vocab_size,
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = 0  # Ensure PAD token is set if the model requires it
    return model
