"""Shortest superpermutation on seven symbols."""
import time
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    if n <= 3:
        # Small cases: use the classic recursive construction
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 4, use the optimized recursive construction with TSP heuristic on permutation order
    # Base: S_3 = 1213121 (length 7)
    # We will build up to n.
    
    # Precompute S_{n-1} and the order of permutations
    # To be efficient, we compute the word for n-1 first.
    
    def get_superperm(k):
        if k == 1:
            return [1]
        if k == 2:
            return [1, 2, 1]
        if k == 3:
            return [1, 2, 1, 3, 1, 2, 1]
            
        # Get S_{k-1}
        prev_word = get_superperm(k - 1)
        
        # Extract permutations of 1..k-1 from prev_word
        perms, seen = [], set()
        for i in range(len(prev_word) - (k - 1) + 1):
            window = tuple(prev_word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        M = len(perms)
        
        # Construct blocks B_i = P_i + [k] + P_i
        blocks = []
        for p in perms:
            block = list(p) + [k] + list(p)
            blocks.append(block)
            
        # Precompute overlap matrix
        # Overlap(i, j) = largest o such that blocks[i][-o:] == blocks[j][:o]
        # Max overlap is k-1 (length of P_i)
        # Actually, it can be up to len(block) - 1, but since blocks are distinct perms, 
        # and they end with P_i and start with P_j, overlap is limited by structure.
        # We just compute it.
        overlap_matrix = [[0] * M for _ in range(M)]
        for i in range(M):
            bi = blocks[i]
            for j in range(M):
                if i == j:
                    continue
                bj = blocks[j]
                # Max possible overlap is min(len(bi), len(bj)) - 1
                # But practically, it's at most k-1 because of the permutation structure
                # Let's just check up to k-1
                max_ov = k - 1
                for o in range(max_ov, 0, -1):
                    if bi[-o:] == bj[:o]:
                        overlap_matrix[i][j] = o
                        break
        
        # TSP Heuristic: Find order of blocks to maximize total overlap
        # Minimize sum of -overlap
        # Start with natural order (0, 1, ..., M-1)
        order = list(range(M))
        
        def calc_cost(ord):
            cost = 0
            for idx in range(len(ord) - 1):
                cost -= overlap_matrix[ord[idx]][ord[idx+1]]
            return cost
        
        current_cost = calc_cost(order)
        
        # Or-Opt heuristic
        # Try to move a segment of length L to a new position
        time_limit = deadline - 20 # Reserve 20s for final steps and validation
        improved = True
        Ls = [1, 2, 3]
        
        while improved and time.time() < time_limit:
            improved = False
            for L in Ls:
                if time.time() > time_limit:
                    break
                for i in range(M - L):
                    if time.time() > time_limit:
                        break
                    # Extract segment
                    seg = order[i:i+L]
                    # Remove segment
                    new_order = order[:i] + order[i+L:]
                    # Find best insertion position
                    best_cost = current_cost
                    best_pos = -1
                    # Calculate cost change
                    # Original cost contributions involving the segment:
                    # Edge (i-1, i), (i+L-1, i+L)
                    # New cost contributions:
                    # Edge (i-1, pos), (pos+L-1, i+L) ... depends on pos
                    
                    # To be fast, we can just recalc cost for the affected parts
                    # Or simply try all positions and pick the best
                    # M is up to 720, so M^2 * L * iters might be heavy.
                    # 720^2 * 3 * 100 = 1.5e8. Might be too slow in Python.
                    
                    # Optimization: 
                    # Cost is sum of -overlap.
                    # Removing seg at i:
                    # We lose overlap(order[i-1], order[i]) and overlap(order[i+L-1], order[i+L])
                    # (with boundaries handled)
                    # Inserting at j:
                    # We gain overlap(new_order[j-1], seg[0]) and overlap(seg[-1], new_order[j])
                    
                    # Let's implement a fast delta calculation.
                    
                    # Get neighbors in original order
                    prev_idx = order[i-1] if i > 0 else None
                    next_idx = order[i+L] if i+L < M else None
                    
                    old_cost_part = 0
                    if prev_idx is not None:
                        old_cost_part -= overlap_matrix[prev_idx][order[i]]
                    if next_idx is not None:
                        old_cost_part -= overlap_matrix[order[i+L-1]][next_idx]
                    
                    # Try inserting at each position j in new_order (length M-L)
                    # new_order length is M-L
                    # Insertion positions 0 to M-L
                    # If insert at j, the segment goes between new_order[j-1] and new_order[j]
                    
                    # We can precompute the best insertion for a given segment?
                    # For now, just loop.
                    
                    for j in range(len(new_order) + 1):
                        # Calculate new cost part
                        new_cost_part = 0
                        if j > 0:
                            new_cost_part -= overlap_matrix[new_order[j-1]][seg[0]]
                        if j < len(new_order):
                            new_cost_part -= overlap_matrix[seg[-1]][new_order[j]]
                        
                        # Total cost change = new_cost_part - old_cost_part
                        delta = new_cost_part - old_cost_part
                        new_total_cost = current_cost + delta
                        
                        if new_total_cost < best_cost:
                            best_cost = new_total_cost
                            best_pos = j
                            
                    if best_pos != -1 and best_cost < current_cost:
                        # Apply move
                        order = new_order[:best_pos] + seg + new_order[best_pos:]
                        current_cost = best_cost
                        improved = True
        
        # Construct final word
        final_word = []
        for idx in order:
            block = blocks[idx]
            if not final_word:
                final_word = block
            else:
                # Find overlap with last block
                last_idx = order[order.index(idx)-1] if idx != order[0] else None
                # Actually, we just need to overlap with the current end of final_word
                # which is the end of the previous block.
                # We can just use the overlap_matrix between the previous block in 'order' and this one.
                # But final_word might be longer than the block? No, we append block[overlap:].
                
                # Find overlap between final_word and block
                # Since final_word ends with the previous block, and block is the current one,
                # the overlap is exactly overlap_matrix[prev_order_idx][idx].
                
                # We need to track the previous index in the order.
                # Let's just do a simple suffix/prefix check to be safe, or store prev_idx.
                
                # Efficiently: we know the previous block in the 'order' list.
                # Let's reconstruct the loop to keep track.
                pass
        
        # Re-do construction with tracked indices
        final_word = []
        prev_idx = None
        for idx in order:
            block = blocks[idx]
            if not final_word:
                final_word = block
            else:
                ov = overlap_matrix[prev_idx][idx]
                final_word += block[ov:]
            prev_idx = idx
            
        return "".join(map(str, final_word))

    # Call for n
    return get_superperm(n)
