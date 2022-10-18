from TeXid import RobertaTeXid, SeqClassifierTokenizer
import torch
ckpt = "TeXid_model/model_v1"
tokenizer = SeqClassifierTokenizer(ckpt)
model = RobertaTeXid.from_pretrained(ckpt)
# model = RobertaTeXid.from_pretrained('proj_testing')
sentence = "I have been doing the exercise"
batch = tokenizer.tokenize(sentence, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
logits = model(batch)
preds = torch.argmax(logits, dim=-1).item()
tense = tokenizer.convertId2Label(preds)
print(tense)