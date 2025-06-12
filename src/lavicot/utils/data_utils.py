from typing import Dict, List, Tuple, Optional, Callable
from transformers import PreTrainedTokenizer
import torch
import random

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<thinking>
...
</thinking>
<answer>
...
</answer>
"""

XML_COT_ANSWER_FORMAT = """\
<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer>
"""

XML_COT_ONLY_FORMAT = """\
<thinking>
{reasoning}
</thinking>
"""



def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_gsm8k_data_components(data_instance: Dict[str, str]) -> Tuple[str, str, str]:
    question = data_instance["question"]
    reasoning = data_instance["answer"].split("####")[0]
    answer = data_instance["answer"].split("####")[1].strip()
    return question, reasoning, answer

def test_formatted_data_same_as_official_template(tokenizer, question):
    official_prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": question.strip()}
        ], tokenize=False, add_generation_prompt=True)
    my_prompt_text = get_formatted_data(question, None, None)[:-11] # remove <thinking> at the end
    same = my_prompt_text.replace("\n","")==official_prompt_text.replace("\n","")
    print(f"is same: {same}")
    print(f"my_prompt_text: {my_prompt_text}")
    print(f"official_prompt_text: {official_prompt_text}")
    return same
    
def get_formatted_data(question, reasoning, answer, add_im_end=True, close_thinking=True):
    """Format data using OpenAI's ChatML format for training."""
    # Build the conversation using OpenAI's ChatML format
    conversation_text = ""
    
    # Add system message
    conversation_text += f"<|im_start|>system\n{SYSTEM_PROMPT.strip()}\n<|im_end|>\n"
    
    # Add user question
    if question is not None:
        conversation_text += f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
    # Add assistant response
    if reasoning is not None:
        conversation_text += "<|im_start|>assistant\n"
        if answer is not None:
            response = XML_COT_ANSWER_FORMAT.format(reasoning=reasoning.strip(), answer=answer.strip())
        else:
            if close_thinking:
                response = XML_COT_ONLY_FORMAT.format(reasoning=reasoning.strip())
            else:
                # Don't close the thinking tag for incomplete reasoning
                response = f"<thinking>\n{reasoning.strip()}"
        conversation_text += response
        if add_im_end:
            conversation_text += "\n<|im_end|>"
    else:
        # If no reasoning, just add the assistant and thinking tags
        conversation_text += "<|im_start|>assistant\n" + "<thinking>\n"

    return conversation_text

def prepare_batch(
    prep_mode: str,
    data_instances: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    min_proportion: float = 0.3,
    max_proportion: float = 0.7,
    device: str = "cuda",
    data_components_extractor: Callable[[Dict[str, str]], Tuple[str, str, str]] = extract_gsm8k_data_components,
    return_question_token_lengths: bool = False
) -> torch.Tensor:
    """Prepare a batch of data for training.
    
    Args:
        data_instances: List of dictionaries containing 'question' and 'answer' keys
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        min_proportion: Minimum proportion of reasoning to use (0-1)
        max_proportion: Maximum proportion of reasoning to use (0-1)
        device: Device to place tensors on
        data_components_extractor: Function to extract question, reasoning, answer from data
        return_question_token_lengths: If True, return (batch_inputs, question_token_lengths) tuple
        
    Returns:
        batch_inputs tensor, or (batch_inputs, question_token_lengths) tuple if return_question_token_lengths=True
    """
    batch_inputs = []
    question_token_lengths = [] if return_question_token_lengths else None
    # is_same = test_formatted_data_same_as_official_template(tokenizer, data_instances[0]["question"])

    for instance in data_instances:
        # Extract components
        question, reasoning, answer = data_components_extractor(instance)
        
        # Calculate question length if needed (before building full input)
        if return_question_token_lengths:
            question_only_text = get_formatted_data(question, None, None)
            question_tokens = tokenizer(question_only_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            question_token_lengths.append(len(question_tokens))
        
        if prep_mode == "question_only":
            # Only use question
            input_text = get_formatted_data(question, None, None)
        elif prep_mode == "cot_only":
            # Use question and partial reasoning
            # Sample a proportion of reasoning tokens
            proportion = random.uniform(min_proportion, max_proportion)
            num_reasoning_words = int(len(reasoning) * proportion)
            
            # Decode the sampled reasoning tokens back to text
            partial_reasoning = reasoning[:num_reasoning_words]
            
            # Combine text components
            input_text = get_formatted_data(question, partial_reasoning, answer)
        elif prep_mode == "full":
            # Use question and full reasoning
            input_text = get_formatted_data(question, reasoning, answer)
        else:
            raise ValueError(f"Invalid prep mode: {prep_mode}")
        
        # Tokenize the complete text
        input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        # Truncate if necessary
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        batch_inputs.append(input_ids)



        
    # Pad batch inputs
    batch_inputs = torch.nn.utils.rnn.pad_sequence(
        batch_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    batch_inputs = batch_inputs.to(device)
    
    if return_question_token_lengths:
        return batch_inputs, question_token_lengths
    else:
        return batch_inputs