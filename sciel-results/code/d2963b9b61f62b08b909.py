import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # 1. Generate standard superpermutation (classic recursive construction)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    new += block[o:]
                    break
        word = new

    if n < 3:
        return "".join(map(str, word))

    # For small n, we can be more aggressive
    if n <= 4:
        return _optimize_small(word, n, deadline)

    # Precompute permutation encoding for fast lookup
    # Encode each permutation as an integer for O(1) dict lookup
    all_perms = list(permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    
    # Build encoding: map tuple -> index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute a fast lookup: for each position in a window, precompute
    # the contribution to the Lehmer code. We'll use a different approach:
    # precompute all possible window encodings using a direct mapping.
    
    # Faster approach: precompute a table for each possible window
    # Since windows are permutations of 1..n, we can use the tuple directly
    # but that's slow. Instead, use a mixed-radix encoding.
    
    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    
    def encode_window_fast(w):
        """Encode a window of n symbols to a unique integer using Lehmer code."""
        # Use a bit-based approach: for each position, track which elements are used
        # Then compute rank via counting
        remaining = [0] * (n + 1)  # 1-indexed, remaining[i] = 1 if i is still available
        for i in range(1, n + 1):
            remaining[i] = 1
        code = 0
        for i in range(n):
            val = w[i]
            # Count remaining values less than val
            count = 0
            for v in range(1, val):
                if remaining[v]:
                    count += 1
            code = code * (n - i) + count
            remaining[val] = 0
        return code

    # Precompute encoding for all permutations
    perm_encodings = [encode_window_fast(list(p)) for p in all_perms]
    enc_to_idx = {enc: i for i, enc in enumerate(perm_encodings)}
    
    # Build initial counts
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        w = word[i:i+n]
        if len(set(w)) == n:
            enc = encode_window_fast(w)
            idx = enc_to_idx.get(enc)
            if idx is not None:
                counts[idx] += 1

    # Fast deletion test using precomputed encodings
    def try_delete_fast(k):
        """Test if deleting position k is valid. Returns (valid, delta_dict) or (False, None)."""
        j_min = max(0, k - n + 1)
        j_max = min(L - n, k)
        
        # Check old windows containing k
        delta = {}
        for j in range(j_min, j_max + 1):
            w = word[j:j+n]
            if len(set(w)) == n:
                enc = encode_window_fast(w)
                idx = enc_to_idx.get(enc)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) - 1
                    if counts[idx] + delta[idx] <= 0:
                        return False, None
        
        # Check new windows formed at junction
        # After deleting k, windows starting from max(0,k-n+1) to min(L-1-n, k-1)
        j_prime_min = max(0, k - n + 1)
        j_prime_max = min(L - 1 - n, k - 1)
        
        for jp in range(j_prime_min, j_prime_max + 1):
            # New window is word[jp:k] + word[k+1:jp+n+1]
            new_w = word[jp:k] + word[k+1:jp+n+1]
            if len(new_w) == n and len(set(new_w)) == n:
                enc = encode_window_fast(new_w)
                idx = enc_to_idx.get(enc)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) + 1
        
        # Final check
        for idx, d in delta.items():
            if counts[idx] + d <= 0:
                return False, None
        
        return True, delta

    # Fast swap test
    def try_swap_fast(i):
        """Test if swapping positions i and i+1 is valid."""
        j_min = max(0, i - n + 1)
        j_max = min(L - n, i + 1)
        
        delta = {}
        
        # Old windows
        for j in range(j_min, j_max + 1):
            w = word[j:j+n]
            if len(set(w)) == n:
                enc = encode_window_fast(w)
                idx = enc_to_idx.get(enc)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) - 1
        
        # New windows after swap
        for j in range(j_min, j_max + 1):
            w = list(word[j:j+n])
            # Apply swap at positions i and i+1 relative to window start j
            pos1 = i - j
            pos2 = i + 1 - j
            if 0 <= pos1 < n and 0 <= pos2 < n:
                w[pos1], w[pos2] = w[pos2], w[pos1]
            if len(set(w)) == n:
                enc = encode_window_fast(w)
                idx = enc_to_idx.get(enc)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) + 1
        
        # Check
        for idx, d in delta.items():
            if counts[idx] + d <= 0:
                return False, None
        
        return True, delta

    # Main optimization loop with smarter ordering
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        L = len(word)
        
        # Try deletions - use a heuristic: try positions where the symbol
        # appears frequently nearby (more likely to be redundant)
        # First pass: try all positions from right to left
        for k in range(L - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            valid, delta = try_delete_fast(k)
            if valid:
                # Apply deletion
                word = word[:k] + word[k+1:]
                for idx, d in delta.items():
                    counts[idx] += d
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        for i in range(L - 1):
            if time.time() > deadline - 5:
                break
            valid, delta = try_swap_fast(i)
            if valid:
                word[i], word[i+1] = word[i+1], word[i]
                for idx, d in delta.items():
                    counts[idx] += d
                improved = True
                break
    
    return "".join(map(str, word))

def _optimize_small(word, n, deadline):
    """More aggressive optimization for small n where we can be thorough."""
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def build_counts(w):
        c = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                c[perm_to_idx[p]] += 1
        return c
    
    counts = build_counts(word)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        L = len(word)
        
        # Try all deletions
        for k in range(L):
            if time.time() > deadline - 5:
                break
            # Test deletion
            j_min = max(0, k - n + 1)
            j_max = min(L - n, k)
            delta = {}
            valid = True
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                idx = perm_to_idx.get(p)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) - 1
                    if counts[idx] + delta[idx] <= 0:
                        valid = False
                        break
            if not valid:
                continue
            j_prime_min = max(0, k - n + 1)
            j_prime_max = min(L - 1 - n, k - 1)
            for jp in range(j_prime_min, j_prime_max + 1):
                new_w = word[jp:k] + word[k+1:jp+n+1]
                p_new = tuple(new_w)
                idx = perm_to_idx.get(p_new)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) + 1
            for idx, d in delta.items():
                if counts[idx] + d <= 0:
                    valid = False
                    break
            if valid:
                word = word[:k] + word[k+1:]
                for idx, d in delta.items():
                    counts[idx] += d
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        for i in range(L - 1):
            if time.time() > deadline - 5:
                break
            j_min = max(0, i - n + 1)
            j_max = min(L - n, i + 1)
            delta = {}
            valid = True
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                idx = perm_to_idx.get(p)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) - 1
            for j in range(j_min, j_max + 1):
                w = list(word[j:j+n])
                pos1 = i - j
                pos2 = i + 1 - j
                if 0 <= pos1 < n and 0 <= pos2 < n:
                    w[pos1], w[pos2] = w[pos2], w[pos1]
                p_new = tuple(w)
                idx = perm_to_idx.get(p_new)
                if idx is not None:
                    delta[idx] = delta.get(idx, 0) + 1
            for idx, d in delta.items():
                if counts[idx] + d <= 0:
                    valid = False
                    break
            if valid:
                word[i], word[i+1] = word[i+1], word[i]
                for idx, d in delta.items():
                    counts[idx] += d
                improved = True
                break
    
    return "".join(map(str, word))
