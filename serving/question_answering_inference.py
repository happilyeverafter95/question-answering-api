import json
import sys

import requests
from transformers import AutoTokenizer
import tensorflow as tf


def get_qa_response(question: str, context: str) -> str:
    # TODO: parametrize model
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    inputs = tokenizer(question, context, add_special_tokens=True)
    input_ids = inputs['input_ids']
    batch = [dict(inputs)]

    input_data = {'instances': batch}
    r = requests.post('http://localhost:8501/v1/models/bert:predict', data=json.dumps(input_data))
    output = json.loads(r.text)['predictions'][0]
    answer_start = tf.argmax([output['start_logits']], axis=1).numpy()[0]
    answer_end = (tf.argmax([output['end_logits']], axis=1) + 1).numpy()[0]
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
