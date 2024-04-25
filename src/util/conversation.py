import torch
import torch.nn as nn

def apply_chat_template(self, messages):
    chat = ""
    for message in messages:
        chat += f"{self.bos_token}{message['role']}\n{message['content']}{self.eos_token}\n"
    return chat

def preprocess_sample(messages: list[dict]):
    """
    process samples into properly formatted strings ready for tokenization
    (probably override this to fit your data)
    """
    conv = []
    model_name = None
    for i, message in enumerate(messages):
        if i == 0:
            conv.append({
                "role": message["from"],
                "content": message["value"]
            })
        else:
            role = "assistant" if message["from"] == "gpt" else "user"
            name = message["value"].split(":")[0]
            if model_name is None and role == "assistant":
                model_name = name

            content = message["value"][len(name)+2:]
            conv.append({
                "role": f"{role} ({name})",
                "content": f"{content}",
            })

    return apply_chat_template(conv)