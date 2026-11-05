import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Construct the 256-word linear code [17,8,6]
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

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1

    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)

    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)

    if not words_short:
        return [0]

    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    
    base_code = np.array(words_short, dtype=np.int64)
    
    # Precompute N0: words at distance < 6 from 0
    N0 = np.array([v for v in range(2**17) if popcount_table[v] < 6], dtype=np.int64)
    
    def update_forbidden(forbidden_count, w, delta):
        neighbors = w ^ N0
        forbidden_count[neighbors] += delta

    def greedy_extend(code_list, forbidden_count, deadline, max_rounds=50):
        """Repeatedly find all available words and add a maximal subset."""
        for rnd in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            available = np.where(forbidden_count == 0)[0]
            if len(available) == 0:
                break
            
            # Greedily add available words, skipping those blocked by previously added in this round
            added_this_round = 0
            # Sort available by some heuristic? Let's just go through them
            # We need to be careful: adding w blocks w^N0
            # We'll iterate and add if still available
            for w in available:
                if time.time() > deadline - 3.0:
                    break
                if forbidden_count[w] != 0:
                    continue  # got blocked by a previously added word in this round
                # Add w
                code_list.append(int(w))
                update_forbidden(forbidden_count, int(w), 1)
                added_this_round += 1
            
            if added_this_round == 0:
                break
        
        return code_list

    def try_swap(code_list, forbidden_count, deadline):
        """Try to remove one word and add multiple others."""
        if time.time() > deadline - 2.0:
            return code_list, False
        
        improved = False
        n = len(code_list)
        # Try a random subset of words to remove
        indices = np.random.permutation(n)
        max_tries = min(100, n)
        
        for _ in range(max_tries):
            if time.time() > deadline - 2.0:
                break
            
            idx_rm = int(indices[_ % len(indices)])
            w_rm = code_list[idx_rm]
            
            # Temporarily remove
            code_list.pop(idx_rm)
            update_forbidden(forbidden_count, int(w_rm), -1)
            
            # Find newly available words (excluding w_rm itself)
            available = np.where(forbidden_count == 0)[0]
            # Filter out w_rm
            mask = available != w_rm
            available = available[mask]
            
            if len(available) >= 2:
                # Try to add as many as possible from available
                added = 0
                for w in available:
                    if time.time() > deadline - 2.0:
                        break
                    if forbidden_count[int(w)] != 0:
                        continue
                    code_list.append(int(w))
                    update_forbidden(forbidden_count, int(w), 1)
                    added += 1
                
                if added >= 2:
                    improved = True
                    # Don't revert, accept the swap
                    # Continue trying more swaps
                    continue
                else:
                    # Revert: remove the added words
                    for _ in range(added):
                        w_add = code_list.pop()
                        update_forbidden(forbidden_count, int(w_add), -1)
            
            # Revert removal
            code_list.append(w_rm)
            update_forbidden(forbidden_count, int(w_rm), 1)
        
        return code_list, improved

    # Main search loop with multiple restarts
    best_code = None
    best_size = 0
    
    np.random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Start from base code
        code_list = list(base_code)
        forbidden_count = np.zeros(2**17, dtype=np.int32)
        
        # Initialize forbidden_count
        for w in code_list:
            update_forbidden(forbidden_count, int(w), 1)
        
        # Greedy extend
        code_list = greedy_extend(code_list, forbidden_count, deadline)
        
        # Try swaps
        code_list, _ = try_swap(code_list, forbidden_count, deadline)
        
        # Do another greedy pass after swaps
        code_list = greedy_extend(code_list, forbidden_count, deadline)
        
        # Try more swaps
        code_list, _ = try_swap(code_list, forbidden_count, deadline)
        
        if len(code_list) > best_size:
            best_size = len(code_list)
            best_code = list(code_list)
        
        # If we haven't improved much, try a different random seed for next restart
        # But we need to be careful about time
    
    if best_code is None:
        return list(base_code)
    
    return best_code
