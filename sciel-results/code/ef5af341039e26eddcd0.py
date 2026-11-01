"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _max_overlap(a, b):
    """Find maximum k such that suffix of a of length k == prefix of b of length k."""
    max_k = min(len(a), len(b))
    for k in range(max_k, -1, -1):
        if k == 0:
            return 0
        if a[-k:] == b[:k]:
            return k
    return 0

def _build_word_from_order(perms, m):
    """Build the word for step m given an ordering of (m-1)-permutations."""
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        ov = _max_overlap(new, block)
        new += block[ov:]
    return new

def _local_search_order(perms, m, time_limit=2.0):
    """Improve ordering of perms to minimize resulting word length via 2-opt."""
    order = list(perms)
    blocks = [list(p) + [m] + list(p) for p in order]
    
    def total_length(order_idx):
        # Compute total length given ordering of indices
        total = len(blocks[order_idx[0]])
        for i in range(1, len(order_idx)):
            prev = blocks[order_idx[i-1]]
            curr = blocks[order_idx[i]]
            ov = _max_overlap(prev, curr)
            total += len(curr) - ov
        return total
    
    best_order = list(range(len(order)))
    best_len = total_length(best_order)
    start = time.time()
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(len(best_order) - 1):
            if (time.time() - start) >= time_limit:
                break
            for j in range(i + 1, len(best_order)):
                # Try reversing segment [i, j]
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_len = total_length(new_order)
                if new_len < best_len:
                    best_order = new_order
                    best_len = new_len
                    improved = True
    return [perms[i] for i in best_order]

def construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Use local search to find a better ordering
        if len(perms) <= 200:
            time_limit = 1.0
        elif len(perms) <= 1000:
            time_limit = 3.0
        else:
            time_limit = 5.0
        
        improved_perms = _local_search_order(perms, m, time_limit)
        word = _build_word_from_order(improved_perms, m)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
