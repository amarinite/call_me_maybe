# build prompt text (menu + question)
# maybe? <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
# in qwen <|im_end|>, ID 151645) -> logit[EOS]
# turn a (FunctionDef list, prompt text) into a text string that the LLM can be fed