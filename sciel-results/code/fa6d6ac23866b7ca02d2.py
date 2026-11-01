import time
import itertools
import random

def construct(n):
    if n == 1:
        return "1"
    
    # Generate superpermutation for n-1
    if n == 2:
        return "121"
        
    # For n > 2, we build from n-1
    # We need the sequence of permutations of 1..n-1
    # We can recursively get the word, then extract perms
    
    def get_word_and_perms(m):
        if m == 1:
            return "1", [(1,)]
        if m == 2:
            return "121", [(1,2), (2,1)]
        
        prev_word, prev_perms = get_word_and_perms(m-1)
        # Extract perms from prev_word? 
        # Or just use prev_perms? 
        # The standard construction uses the order they appear in the word.
        # prev_perms is already in that order.
        
        # Now build word for m
        # Overlap matrix for perms of m-1
        # Actually, we just need to build the word for m to pass to m+1?
        # No, we only need the list of perms for the final step optimization.
        # But to get the perms for m, we need to know the order.
        # The order is determined by the construction.
        
        # Let's just build the word for n-1 normally.
        # Then extract the perms.
        
        # This recursive extraction is slow if done naively.
        # But n is small.
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        # Extract unique perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return "".join(map(str, new_word)), perms

    # Get the word and perms for n-1
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s for final construction
    
    _, perms = get_word_and_perms(n-1)
    num_perms = len(perms)
    
    # Map perms to indices
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Max k is n-2 (since they are distinct perms of n-1)
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Check overlaps
            # pi suffix vs pj prefix
            # Max possible overlap is n-2
            for k in range(max_k, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Initial order: 0 to num_perms-1
    order = list(range(num_perms))
    
    def calc_length(ord_list):
        total = 0
        # First block length is 2*(n-1) + 1 = 2n-1
        total += 2 * (n - 1) + 1
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
        return total
    
    best_length = calc_length(order)
    best_order = order[:]
    
    # Local Search
    # Swap adjacent
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(order) - 1):
            # Try swap i, i+1
            order[i], order[i+1] = order[i+1], order[i]
            new_len = calc_length(order)
            if new_len < best_length:
                best_length = new_len
                best_order = order[:]
                improved = True
            else:
                # Revert
                order[i], order[i+1] = order[i+1], order[i]
                
    # If time remains, try 2-opt or random restarts?
    # For now, simple adjacent swap might be enough to find small improvements.
    
    # Construct final string
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
