from transformers import AutoTokenizer

class SeqClassifierTokenizer:
    def __init__(self, tokenizer_ck: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ck)
        self.label2id = {
            'Future Continuous': 0,
            'Past Continuous': 1,
            'Past Simple': 2,
            'Present Continuous': 3,
            'Present Perfect': 4,
            'Present Perfect Continuous': 5,
            'Present Simple': 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def tokenize(self, sentences, **kwargs):
        return self.tokenizer(sentences, **kwargs)

    def convertLabel2Id(self, label):
        return self.label2id[label]

    def convertId2Label(self, id):
        return self.id2label[id]