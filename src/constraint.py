# token masking, argmax, choice among strings

# logits     →  list of ~150k floats
# allowed    →  set of ints (which IDs are OK right now)
# masked     →  logits where non-allowed entries are -inf
# next_id    →  argmax(masked)  →  always a legal token
# turn (generated text, schema context) into (allowed token IDs, is_complete)

import numpy as np

def mask_logits(logits: np.ndarray, allowed: set[int]) -> np.ndarray:
	"""Return a copy of `logits` with all non-allowed indices set to -inf."""
	if not allowed:
		raise ValueError("empty `allowed` set: no legal next token")
	masked = np.full_like(logits, -np.inf)
	allowed_idx = np.array(sorted(allowed), dtype=int)
	masked[allowed_idx] = logits[allowed_idx]
	return masked


def argmax(logits: np.ndarray) -> int:
	"""Return the index of the maximum element."""
	return int(logits.argmax())


# TEMP
def constrain_choice(
	candidates: list[str], prefix: str, vocab: dict[int, str]
	) -> tuple[set[int], bool]:
	"""Return (allowed_ids, is_complete) for the 'choose one of candidates' state."""
	is_complete = prefix in candidates

	allowed_ids: set[int] = set()

	if is_complete:
		pass
	
	for token_id, token_next in vocab.items():
		candidate_str = prefix + token_next
		if any(c.startswith(candidate_str) for c in candidates):
			allowed_ids.add(token_id)
	
	return allowed_ids, is_complete

# The legal tokens from a given prefix are: 
# all tokens whose decoded text takes you from the current 
# trie node to some descendant node without leaving the trie.
# stop when is_complete and (not allowed_ids or the model's top preference among allowed_ids is worse than some baseline)
# 1. Mask and argmax over legal tokens → best_token_id
#		If is_complete, use logit[EOS] as the "stop" preference.
#		If logit[EOS] > logit[best_legal_continuation], stop.

# rng = np.random.default_rng(seed=40)
# logits = rng.standard_normal(10)

# allowed = {2, 5, 7}

# print("logits:", np.round(logits, 2))

# masked = mask_logits(logits, allowed)
# best = argmax(masked)

# print(f"--- allowed={allowed} ---")
# print("  masked:  ", np.round(masked, 2))
# print(f"  argmax:   {best}")
# assert best in allowed, f"BUG: argmax {best} not in {allowed}"
# assert masked[best] == logits[best]
# print(f"  score:    {logits[best]:.3f}")