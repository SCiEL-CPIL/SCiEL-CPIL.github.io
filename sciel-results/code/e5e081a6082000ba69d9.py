import numpy as np
import time

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 85.0

def _qr_code():
    """[18,9,6] extended quadratic-residue code (256 words) shortened to length 17."""
    p = 17
    QR = set((i * i) % p for i in range(1, p))
    G = np.zeros((8, p), dtype=np.uint8)
    for k in range(8):
        for i in range(p):
            if i == 0:
                G[k, i] = 1
            elif (i % p) in QR:
                G[k, i] = (k + 1) % 2
    par = np.array([G[:, i].sum() % 2 for i in range(p)], dtype=np.uint8)
    full = np.zeros((256, p + 1), dtype=np.uint8)
    for m in range(256):
        bits = np.array([(m >> j) & 1 for j in range(8)], dtype=np.uint8)
        row = (bits @ G) % 2
        full[m, :p] = row
        full[m, p] = row.sum() % 2
    mask = full[:, p] == 0
    short = full[mask, :p]
    powers = (1 << np.arange(p, dtype=np.int64))[::-1]
    ints = (short @ powers).astype(np.int64)
    return ints

def construct():
    t0 = time.time()
    deadline = t0 + (_DEADLINE - (time.time() - _START))
    if deadline - t0 < 1.0:
        deadline = t0 + 1.0

    # 1. Generate all 2^17 words
    all_words = np.arange(1 << _N, dtype=np.int64)
    
    # 2. Precompute bit matrix for all words: shape (2^17, 17)
    # We want W_bits[i, j] to be the j-th bit of all_words[i]
    # Let's define bit 0 as LSB.
    bit_masks = 1 << np.arange(_N, dtype=np.int64)
    # W_bits: (N_total, 17)
    # all_words[:, None] ^ bit_masks[None, :] doesn't work directly for bits.
    # Use: (all_words[:, None] & bit_masks[None, :]) > 0
    W_bits = (all_words[:, None] & bit_masks[None, :]) > 0
    W_bits = W_bits.astype(np.uint8)
    
    # 3. Get seed code
    try:
        seed_ints = _qr_code()
        seed_ints = np.unique(seed_ints)
    except Exception:
        seed_ints = np.array([0], dtype=np.int64)
        
    code_list = seed_ints.tolist()
    code_set = set(code_list)
    
    # 4. Initialize min_dist array
    # min_dist[i] = min hamming distance between all_words[i] and any word in code_list
    min_dist = np.full(len(all_words), 100, dtype=np.int32)
    
    # Update min_dist for all seed words
    for c in seed_ints:
        # Get bits of c
        c_bits = (c & bit_masks) > 0
        c_bits = c_bits.astype(np.uint8)
        # Compute distance to all words
        # dist = sum(W_bits ^ c_bits, axis=1)
        d = np.sum(W_bits ^ c_bits, axis=1)
        min_dist = np.minimum(min_dist, d)
        
    # 5. Greedy Max-Min Loop
    # We keep adding the word with the largest min_dist, as long as it is >= _MIN_D
    while time.time() < deadline - 1.0:
        # Find the index with the maximum min_dist
        idx = np.argmax(min_dist)
        max_d = min_dist[idx]
        
        if max_d < _MIN_D:
            break
            
        # Get the word
        cand = int(all_words[idx])
        
        # Add it
        code_list.append(cand)
        code_set.add(cand)
        
        # Update min_dist for the new word
        # We already have W_bits, so we can get cand_bits from W_bits[idx]
        cand_bits = W_bits[idx]
        d = np.sum(W_bits ^ cand_bits, axis=1)
        min_dist = np.minimum(min_dist, d)
        
    return code_list

def main():
    return construct()
