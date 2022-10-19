from TeXid import PostProcess
import torch
ckpt = "TeXid_model/model_v4"
postprocess = PostProcess(ckpt)
sents = [
    "I will be there",
    "I will be working over there",
    "I have joined the project",
    "I work as a teacher",
    "I have been working there for 2 years and a half",
    "I was at home",
    "I was dealing with this problems",
    "I ran into trouble yesterday",
    "I am performing a research",
]
for sent in sents:
    print(postprocess(sent))