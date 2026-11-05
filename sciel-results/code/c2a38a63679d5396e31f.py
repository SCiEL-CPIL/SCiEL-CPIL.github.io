"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols using the standard
    recursive construction, with an optimization pass for n=7 to reorder
    the permutations of 1..6 to minimize the total length.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        # Optimization for m=7 (adding 7 to perms of 1..6)
        if m == 7:
            # We have 720 permutations.
            # We want to find an order that maximizes total overlap.
            # Overlap between prev and curr is longest suffix of prev matching prefix of curr.
            # prev is a tuple of 6 elements. curr is a tuple of 6 elements.
            
            # Precompute overlaps
            # perms is a list of tuples.
            # Map tuple to index for fast lookup?
            # Or just compute on the fly if 720 is small enough.
            # 720^2 = 518400 pairs. Computing overlap is O(6).
            # Total ops ~ 3M. This is fast.
            
            # Let's define the cost function.
            # We start with the order from the standard construction (current 'perms' list).
            # Or we can start with a random order and optimize.
            # The standard order is likely close to optimal or at least a good start.
            
            order = list(range(len(perms)))
            
            # Calculate current score (total overlap)
            def calc_total_overlap(order):
                total = 0
                prev = tuple(word[-6:]) # The end of the current word (S_6)
                for idx in order:
                    curr = perms[idx]
                    # Calculate overlap
                    max_ov = 0
                    for k in range(6, 0, -1):
                        if prev[-k:] == curr[:k]:
                            max_ov = k
                            break
                    total += max_ov
                    prev = curr
                return total
            
            current_score = calc_total_overlap(order)
            
            # 2-Opt / Local Search
            # We try to swap adjacent elements or reverse segments.
            # Given the time limit, we can do a few passes.
            improved = True
            max_iterations = 200 # Adjust based on time
            iter_count = 0
            while improved and iter_count < max_iterations and time.time() < deadline - 5:
                improved = False
                iter_count += 1
                for i in range(len(order) - 1):
                    if time.time() > deadline - 5:
                        break
                    # Try swapping i and i+1
                    old_score = current_score
                    order[i], order[i+1] = order[i+1], order[i]
                    new_score = calc_total_overlap(order)
                    if new_score > old_score:
                        current_score = new_score
                        improved = True
                    else:
                        order[i], order[i+1] = order[i+1], order[i]
                    
                    # Try reversing a small segment? 
                    # 2-opt on the path: reverse order[i:j]
                    # This might be expensive to recalculate fully.
                    # Let's stick to adjacent swaps for now, or simple 2-opt.
                    # Actually, 2-opt reversal is better.
                    # But adjacent swap is O(N) to update score if we do it smartly.
                    # Here calc_total_overlap is O(N).
                    # So 2-opt pass is O(N^2).
                    # 720^2 = 518k. 200 iterations = 100M ops. Might be slow in Python.
                    
                    # Let's optimize the score calculation.
                    # We can calculate the change in score for a swap.
                    # Score = sum(overlap(prev, curr)).
                    # Swap i, i+1.
                    # Old: ... A, B, C ...
                    # New: ... A, C, B ...
                    # Change: -overlap(A,B) - overlap(B,C) + overlap(A,C) + overlap(C,B)
                    # Where A = order[i-1] (or start), B = order[i], C = order[i+1].
                    
                    # Let's implement incremental score update.
                    pass # We will rewrite the loop below
                    
                # If no improvement in a full pass, break?
                # Or do a few random restarts?
                # Given the small gap (8 symbols), local search might get stuck.
                # But let's try.
                
            # Re-implement local search with incremental updates
            order = list(range(len(perms)))
            current_score = calc_total_overlap(order)
            
            def get_overlap(p1, p2):
                # p1, p2 are tuples
                for k in range(6, 0, -1):
                    if p1[-k:] == p2[:k]:
                        return k
                return 0
            
            # Precompute overlaps for all pairs?
            # 720x720 matrix of bytes/ints.
            # 518k entries.
            # Memory is fine. Time to compute is fine.
            # Let's do it to speed up local search.
            ov_matrix = [[0]*len(perms) for _ in range(len(perms))]
            # Also need overlap from start (end of S_6) to each perm.
            start_ov = [0]*len(perms)
            start_end = tuple(word[-6:])
            for i in range(len(perms)):
                start_ov[i] = get_overlap(start_end, perms[i])
            
            for i in range(len(perms)):
                for j in range(len(perms)):
                    if i != j:
                        ov_matrix[i][j] = get_overlap(perms[i], perms[j])
            
            # Now local search on order
            # Score = start_ov[order[0]] + sum(ov_matrix[order[i]][order[i+1]])
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(order) - 1):
                    if time.time() > deadline - 5:
                        break
                    # Swap i and i+1
                    a = order[i-1] if i > 0 else -1
                    b = order[i]
                    c = order[i+1]
                    d = order[i+2] if i+2 < len(order) else -1
                    
                    # Old contribution
                    old_contrib = 0
                    if a != -1:
                        old_contrib += ov_matrix[a][b]
                    else:
                        old_contrib += start_ov[b]
                    old_contrib += ov_matrix[b][c]
                    if d != -1:
                        old_contrib += ov_matrix[c][d]
                    
                    # New contribution
                    new_contrib = 0
                    if a != -1:
                        new_contrib += ov_matrix[a][c]
                    else:
                        new_contrib += start_ov[c]
                    new_contrib += ov_matrix[c][b]
                    if d != -1:
                        new_contrib += ov_matrix[b][d]
                    
                    if new_contrib > old_contrib:
                        order[i], order[i+1] = order[i+1], order[i]
                        current_score += new_contrib - old_contrib
                        improved = True
            
            # Use the optimized order
            for idx in order:
                p = perms[idx]
                block = list(p) + [m] + list(p)
                
                # Find overlap
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        else:
            # Standard construction for m < 7
            for p in perms:
                block = list(p) + [m] + list(p)
                
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
