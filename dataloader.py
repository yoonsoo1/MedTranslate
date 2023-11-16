from torch.utils.data import Dataset

class TextSimplificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = "$simple$ ; $expert$ = " + self.texts[idx]
        label = "$simple$ = " + self.labels[idx]

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text, 
            None, 
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Tokenize the label
        labels = self.tokenizer.encode_plus(
            label,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten(),
            'labels_attention_mask': labels['attention_mask'].flatten()
        }