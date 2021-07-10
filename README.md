# question-answering-api

Adapted the tf-serving example from [this Huggingface blog post](https://huggingface.co/blog/tf-serving) for question-answering.

Run `./run_server.sh` to expose a REST API for inferencing using TensorFlow-Serving. The API endpoint requires tokenized inputs, so some preprocessing is needed for inferencing with raw text.

For inferencing, see [question_answering_inference.py](serving/question_answering_inference.py). Example usage:

```
context = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

get_qa_response('how many pretrained models are there?', context)
```