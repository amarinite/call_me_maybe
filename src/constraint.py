# JSON-schema state machine + token masking

# logits     →  list of ~150k floats
# allowed    →  set of ints (which IDs are OK right now)
# masked     →  logits where non-allowed entries are -inf
# next_id    →  argmax(masked)  →  always a legal token

import numpy as np

def mask_logits(logits: list[float], allowed: set[int]) -> list[float]:
	"""Return a copy of `logits` with all non-allowed indices set to -inf."""
	masked = np.full_like(logits, -np.inf)
	allowed_idx = np.array(sorted(allowed), dtype=int)
	masked[allowed_idx] = logits[allowed_idx]
	return masked

def argmax(logits: list[float]) -> int:
	"""Return the index of the maximum element."""
	return int(logits.argmax())


rng = np.random.default_rng(seed=40)
logits = rng.standard_normal(10)

allowed = {2, 5, 7}


print("logits:", np.round(logits, 2))

masked = mask_logits(logits, allowed=set())
best = argmax(masked)

print(f"--- allowed={allowed} ---")
print("  masked:  ", np.round(masked, 2))
print(f"  argmax:   {best}")
assert best in allowed, f"BUG: argmax {best} not in {allowed}"
assert masked[best] == logits[best]
print(f"  score:    {logits[best]:.3f}")