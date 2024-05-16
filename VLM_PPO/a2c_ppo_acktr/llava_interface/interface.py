import torch
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def llava_generate(value_model, tokenizer, input_ids, image_tensor, args):
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype = base.dtype)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(input_ids.to(base.device), None, None, None, None, image_tensor)
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    with torch.inference_mode():
        outputs = base.generate(
        inputs_embeds = inputs_embeds,
        do_sample=True,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,)
        output_ids = outputs['sequences']
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids
    with torch.no_grad():
        values, sum_log_probs, action_tokens_log_prob = llava_evaluate(value_model, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef)
    return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def llava_evaluate(value_model, input_ids, output_ids, image_tensor, temperature, thought_prob_coef):
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(torch.cat([input_ids, output_ids], dim = 1), None, None, None, None, image_tensor)

    #calling the model
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    #omit the first output token
    outputs = base(
        inputs_embeds = inputs_embeds,
        output_hidden_states = True,
        )
    scores = outputs.logits

    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    values = value_model.value_head(hidden_states)
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    output_ids_mask = (output_ids != 0)[:, 1:]
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    target = torch.tensor([29908,2467,1115]).to(base.device)
    # tokens for text string:'"action":' (torch.tensor([[29908,2467,1115]]))
    matches = (unfolded == target).all(dim = -1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    if match_index.shape[0] > 1:
        match_index = match_index[-1].unsqueeze(0)
    else:
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob
