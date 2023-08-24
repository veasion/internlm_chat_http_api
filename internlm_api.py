import json
import torch
import torch.nn as nn
from typing import List, Callable, Optional
import copy
import warnings
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')
logging.getLogger().setLevel(logging.INFO)

# model: internlm-chat-7b, internlm-chat-7b-8k, internlm-chat-7b-v1_1
# model_path = "/root/autodl-tmp/internlm-chat-7b-8k"
model_path = "internlm-chat-7b-8k"

# load model
torch.cuda.empty_cache()
print(f'load model {model_path} begin.')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'load model {model_path} end.')


@torch.inference_mode()
def generate_interactive(
        prompt,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        additional_eos_token_id: Optional[int] = None,
        **kwargs
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            warnings.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        warnings.warn(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


# system_prompt = "<|System|>:{system}<TOKENS_UNUSED_2>\n"
system_prompt = "<|System|>:{system}<eos>\n"
user_prompt = "<|User|>:{user}<eoh>\n"
robot_prompt = "<|Bot|>:{assistant}<eoa>\n"
# cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


def combine_prompt(messages):
    total_prompt = ""
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "system":
            cur_prompt = system_prompt.replace("{system}", cur_content)
        elif message["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "assistant":
            cur_prompt = robot_prompt.replace("{assistant}", cur_content)
        else:
            raise RuntimeError("role error: " + message["role"])
        total_prompt += cur_prompt
    return total_prompt + '<|Bot|>:'


def response_stream(response, debug=False):
    last_str = None
    for r in response:
        if debug:
            logging.info(r)
        response_str = r
        if last_str and response_str.startswith(last_str):
            new_response_str = response_str[len(last_str):]
            last_str = response_str
            response_str = new_response_str
        else:
            last_str = response_str
        yield 'data: ' + json.dumps({'content': response_str}) + '\n\n'
    yield 'data: {"is_end":true}'


@app.route('/internlm/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data['messages']
        debug = 'debug' in data and data['debug']
        stream = 'stream' in data and data['stream']

        prompt = combine_prompt(messages)

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.top_p = data['top_p'] if 'top_p' in data else 0.8 # 0-1
        generation_config.do_sample = data['do_sample'] if 'do_sample' in data else True
        generation_config.temperature = data['temperature'] if 'temperature' in data else 0.8  # 0-1
        generation_config.max_new_tokens = data['max_tokens'] if 'max_tokens' in data else 2048 # 20-8196
        generation_config.repetition_penalty = data['repetition_penalty'] if 'repetition_penalty' in data else 1.0

        response = generate_interactive(prompt=prompt, generation_config=generation_config, additional_eos_token_id=103028)

        if stream:
            return Response(response_stream(response, debug), mimetype='text/event-stream')
        else:
            response_str = ''
            for r in response:
                response_str = r
            if debug:
                logging.info(response_str)
            res = {
                'code': 0,
                'data': { 'content': response_str },
                'message': 'success'
            }
    except Exception as e:
        logging.exception(f'request error: \n{e}')
        res = {'code': -1, 'message': f'error: {str(e)}'}
    return jsonify(res)


if __name__ == "__main__":
    # web html page: streamlit run web_demo.py --server.port=6006
    logging.info('start flask')
    from waitress import serve
    # serve(app, host='0.0.0.0', port=6006)
    serve(app, host='0.0.0.0', port=8002)
