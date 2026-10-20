import json
from pathlib import Path

import numpy as np

from llm_sdk import Small_LLM_Model


def main() -> None:
    print("Loading model (first run may take a while)...")
    model = Small_LLM_Model()
    print("Model loaded.")

    # encode
    text = "What is the sum of 213 and 3?"
    tensor = model.encode(text)
    print(f"\nencode({text!r})")
    print(f"  type:  {type(tensor)}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  raw:   {tensor}")

    input_ids = tensor[0].tolist()
    print(f"  list[int]: {input_ids}  (len={len(input_ids)})")

    # vocab
    vocab_path = Path(model.get_path_to_vocab_file())
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    print(f"\nvocab file: {vocab_path}")
    print(f"  entries: {len(raw_vocab)}")
    print(f"  first 3 raw items: {list(raw_vocab.items())[:3]}")

    first_key = next(iter(raw_vocab))
    if isinstance(first_key, str) and first_key.isdigit():
        id_to_token = {int(k): v for k, v in raw_vocab.items()}
        print("  direction: ID -> token")
    else:
        id_to_token = {int(v): k for k, v in raw_vocab.items()}
        print("  direction: token -> ID (inverted)")

    # logits
    logits = model.get_logits_from_input_ids(input_ids)
    print(f"\nget_logits_from_input_ids")
    print(f"  type: {type(logits)}")
    print(f"  len:  {len(logits)}  (vocab size: {len(id_to_token)})")
    print(f"  match: {len(logits) == len(id_to_token)}")

    # argmax + decode
    arr = np.asarray(logits)
    best_id = int(arr.argmax())
    best_logit = float(arr[best_id])
    print(f"\nargmax")
    print(f"  id:    {best_id}")
    print(f"  logit: {best_logit:.4f}")
    print(f"  token: {id_to_token.get(best_id)!r}")
    print(f"  decode: {model.decode([best_id])!r}")

    # top 10
    print(f"\ntop 10 tokens next")
    top10_idx = np.argsort(arr)[-10:][::-1]
    for idx in top10_idx:
        i = int(idx)
        print(
            f"  id={i:>6}  logit={float(arr[i]):>8.3f}  "
            f"token={id_to_token.get(i)!r}  "
            f"decode={model.decode([i])!r}"
        )

    # max min
    print(f"\nmax token: {max(id_to_token)}")
    print(f"\nmin token: {min(id_to_token)}")


if __name__ == "__main__":
    main()
