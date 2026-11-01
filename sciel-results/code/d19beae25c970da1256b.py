"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _max_overlap_fast(a, b):
    """Find maximum k such that suffix of a of length k == prefix of b of length k."""
    max_k = min(len(a), len(b))
    # Check from largest possible overlap down
    for k in range(max_k, 0, -1):
        # Quick check: compare last k of a with first k of b
        match = True
        for i in range(k):
            if a[len(a)-k+i] != b[i]:
                match = False
                break
        if match:
            return k
    return 0

def _build_word_from_order(perms, m):
    """Build the word for step m given an ordering of (m-1)-permutations."""
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        ov = _max_overlap_fast(new, block)
        new += block[ov:]
    return new

def _local_search_order(perms, m, time_limit=2.0):
    """Improve ordering of perms to minimize resulting word length via 2-opt with incremental cost."""
    n_perms = len(perms)
    blocks = [list(p) + [m] + list(p) for p in perms]
    block_lens = [len(b) for b in blocks]
    
    # Precompute overlap matrix
    ov_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                ov_matrix[i][j] = _max_overlap_fast(blocks[i], blocks[j])
    
    def get_cost(order_idx):
        """Compute total length for given ordering."""
        if len(order_idx) == 0:
            return 0
        total = block_lens[order_idx[0]]
        for i in range(1, len(order_idx)):
            prev = order_idx[i-1]
            curr = order_idx[i]
            total += block_lens[curr] - ov_matrix[prev][curr]
        return total
    
    best_order = list(range(n_perms))
    best_len = get_cost(best_order)
    start = time.time()
    deadline = start + time_limit
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(n_perms - 1):
            if (time.time() - start) >= deadline:
                break
            for j in range(i + 1, n_perms):
                # Compute cost delta for reversing segment [i, j]
                # Old: ... A[i-1] -> A[i] -> ... -> A[j] -> A[j+1] ...
                # New: ... A[i-1] -> A[j] -> ... -> A[i] -> A[j+1] ...
                
                old_cost = 0
                new_cost = 0
                
                # Edge before segment
                if i > 0:
                    old_cost += block_lens[best_order[i]] - ov_matrix[best_order[i-1]][best_order[i]]
                    new_cost += block_lens[best_order[j]] - ov_matrix[best_order[i-1]][best_order[j]]
                
                # Internal edges
                for k in range(i, j):
                    old_cost += block_lens[best_order[k+1]] - ov_matrix[best_order[k]][best_order[k+1]]
                    new_cost += block_lens[best_order[k]] - ov_matrix[best_order[k+1]][best_order[k]]
                
                # Edge after segment
                if j < n_perms - 1:
                    old_cost += block_lens[best_order[j+1]] - ov_matrix[best_order[j]][best_order[j+1]]
                    new_cost += block_lens[best_order[j+1]] - ov_matrix[best_order[i]][best_order[j+1]]
                
                delta = new_cost - old_cost
                if delta < 0:
                    # Apply the reversal
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_len += delta
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
        if len(perms) <= 100:
            time_limit = 1.0
        elif len(perms) <= 500:
            time_limit = 3.0
        elif len(perms) <= 2000:
            time_limit = 8.0
        else:
            time_limit = 15.0
        
        improved_perms = _local_search_order(perms, m, time_limit)
        word = _build_word_from_order(improved_perms, m)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
