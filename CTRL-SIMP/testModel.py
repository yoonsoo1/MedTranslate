from transformers import T5Tokenizer, T5ForConditionalGeneration

input_string = "<elab>"
tokenizer = T5Tokenizer.from_pretrained("t5-large")
input_ids = tokenizer.encode(input_string)
print(input_ids)
