"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _build_linear_code():
    """Construct the 256-word [17,8,6] code from the [18,9,6] extended QR code.
    
    Uses GF(2^8) arithmetic to build the generator polynomial, then constructs
    the [17,9,5] QR code, extends to [18,9,6] via parity, and shortens by one
    coordinate to get [17,8,6] with 256 words.
    """
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x1D)
    poly = 0x1D
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
    
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
    
    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
    
    if alpha is None:
        # Fallback: return a simple code
        return [0]
    
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # The QR code of length 17 has generator polynomial with roots at
    # alpha^i for i in the quadratic residues
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) = prod_{i in Q} (x - alpha^i)
    # Start with g(x) = 1
    g_coeffs = [1]  # g(x) = 1, coefficients from x^0 upward
    
    for i in Q:
        root = gf_pow(alpha, i)
        # Multiply g(x) by (x - root) = (x + root) over GF(2)
        # new_g[j] = old_g[j-1] + root * old_g[j]
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
    
    # g_coeffs has length 9 (degree 8), so the code is [17, 9, 5]
    # Build generator matrix: cyclic shifts of g_coeffs
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
    
    # Generate all 512 codewords of the [17,9,5] code
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
    
    # Extend to length 18 by adding parity bit
    words_18 = []
    for w in words_17:
        parity = int(np.sum(w) % 2)
        word_ext = np.append(w, parity)
        words_18.append(word_ext)
    
    # Shorten by coordinate 17 (the parity bit): keep words where bit 17 = 0
    # This gives a [17,8,6] code with 256 words
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
    
    return words_short


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6.
    
    Uses a nonlinear greedy extension starting from the 256-word linear code,
    with multiple random restarts and a precomputed popcount table for speed.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        # Fallback: use a simple construction
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    
    # Precompute popcount table for all 17-bit values
    popcount_table = np.array([i.bit_count() for i in range(2**N)], dtype=np.int8)
    
    # Greedy extension with multiple restarts
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        
        # Create a shuffled list of candidates
        rng = np.random.default_rng(seed=restart * 1000 + 42)
        candidates = rng.permutation(2**N)
        
        current_list = list(best_words)
        current_set = set(current_list)
        current_arr = np.array(current_list, dtype=np.int64)
        
        # Try adding each candidate
        for w in candidates:
            if time.time() > deadline - 2.0:
                break
            
            if w in current_set:
                continue
            
            # Vectorized distance check
            diffs = current_arr ^ w
            dists = popcount_table[diffs]
            min_d = int(np.min(dists))
            
            if min_d >= MIN_DIST:
                current_list.append(w)
                current_set.add(w)
                current_arr = np.append(current_arr, w)
        
        # Update best if improved
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_words = current_list
    
    return best_words

# EVOLVE-BLOCK-END
