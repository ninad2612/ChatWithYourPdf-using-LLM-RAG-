from transformers import AutoTokenizer, AutoModel

model_name = 'sentence-transformers/all-MiniLM-L12-v2'

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained(r'your-path')
model.save_pretrained(r'your-path')