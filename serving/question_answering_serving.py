import os
import json

import requests
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf


MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'


def create_saved_model() -> None:
    model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL)
    model.save_pretrained('model', saved_model=True)


def get_qa_response(question: str, context: str) -> str:
    if not os.path.exists('model'):
        create_saved_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(question, context, add_special_tokens=True)
    input_ids = inputs['input_ids']
    batch = [dict(inputs)]

    input_data = {'instances': batch}
    r = requests.post('http://localhost:8501/v1/models/bert:predict', data=json.dumps(input_data))
    output = json.loads(r.text)['predictions'][0]
    answer_start = tf.argmax([output['start_logits']], axis=1).numpy()[0]
    answer_end = (tf.argmax([output['end_logits']], axis=1) + 1).numpy()[0]
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
