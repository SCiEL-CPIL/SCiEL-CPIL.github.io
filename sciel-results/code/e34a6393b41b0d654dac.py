"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general small cases if possible, or specialize.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # The task is specifically K_3(6,1).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters if they appear, 
        # though the contract implies this specific task.
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Very small case, just return all words or brute force
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specialized solver for K_3(6,1)
    N = 3 ** n  # 729
    
    # Precompute balls: ball[word_idx] is a boolean array of size N or a set of indices
    # Using numpy bitsets or boolean arrays is efficient.
    # word_idx -> tuple of digits
    # We can map index to digits and back.
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    # Precompute all balls
    # balls[i] is a boolean array of length N where balls[i][j] is True if dist(i, j) <= radius
    balls = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        wi = idx_to_word(i)
        # Center
        balls[i, i] = True
        # Distance 1 neighbors
        for pos in range(n):
            orig_val = wi[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                wj = wi[:]
                wj[pos] = new_val
                j = word_to_idx(wj)
                balls[i, j] = True
        # Radius is 1, so no distance 2 needed.

    # Greedy Set Cover
    # uncovered: boolean array, True if not covered
    uncovered = np.ones(N, dtype=bool)
    code_indices = []
    
    # Precompute coverage counts? 
    # Greedy step: pick i that maximizes sum(balls[i] & uncovered)
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Calculate coverage of each candidate
        # To speed up, we can skip candidates that are already in code? 
        # No, we can pick any word.
        # But picking a word already in code is useless.
        
        # Optimization: Only consider words not in code?
        # Actually, if a word is in code, its ball is already added. 
        # So we should only pick new words.
        
        # For N=729, checking all 729 candidates is fast.
        best_idx = -1
        best_count = -1
        
        # We can vectorize the counting
        # covered_by_i = balls[i] & uncovered
        # count = sum(covered_by_i)
        
        # Loop is fine for N=729
        for i in range(N):
            if i in code_indices:
                continue
            # Count how many currently uncovered words are covered by i
            # This is sum(balls[i] & uncovered)
            # Using numpy:
            cnt = np.sum(balls[i] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        uncovered[balls[best_idx]] = False
        
        if best_count == 0:
            break

    current_code = list(code_indices)
    
    # Local Search
    # 1. Try to remove a codeword if its ball is covered by others.
    # 2. Try to swap a codeword with another to improve coverage (reduce size? No, swap keeps size same, but might enable removal later).
    # Actually, standard local search for minimum dominating set:
    # - Check if any vertex in D is redundant. Remove it.
    # - Check if any vertex not in D can replace a vertex in D such that the new set is still dominating? 
    #   This is hard. 
    #   Better: 
    #   1. Remove redundant vertices.
    #   2. Try to add a vertex and then remove two redundant vertices? (2-opt)
    #   Or simpler: 
    #   1. Remove redundant.
    #   2. For each c in D, and each c' not in D, if (D - {c} + {c'}) is dominating, do it? 
    #      This doesn't reduce size. 
    #      We want to reduce size. 
    #      So: If (D - {c}) is NOT dominating, but (D - {c, d} + {c'}) IS dominating?
    
    # Let's stick to:
    # 1. Greedy Removal: 
    #    While True:
    #      found_removal = False
    #      for c in current_code:
    #         if not any(uncovered_word in ball(c) for uncovered_word in ... wait)
    #         Check if ball(c) is subset of union(ball(d) for d in current_code if d != c)
    #         If yes, remove c.
    #
    # 2. Greedy Swap (to enable removal):
    #    For each c in current_code:
    #       For each c' not in current_code:
    #          If (current_code - {c} + {c'}) is dominating:
    #             Do the swap? 
    #             This doesn't reduce size immediately.
    #             But it might change the structure to allow a removal.
    
    # Given N=729, we can afford to check dominance of a set of size ~75.
    
    def is_dominating(code_set):
        # code_set is a list of indices
        if not code_set:
            return N == 0
        # Union of balls
        # We can use a boolean array
        covered = np.zeros(N, dtype=bool)
        for c in code_set:
            covered |= balls[c]
        return np.all(covered)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove
        for i in range(len(current_code)):
            if time.time() > deadline: break
            c = current_code[i]
            # Check if c is redundant
            # If removing c leaves the set dominating
            new_code = current_code[:i] + current_code[i+1:]
            if is_dominating(new_code):
                current_code = new_code
                improved = True
                break # Restart loop as indices changed
                
        if improved:
            continue
            
        # 2. Try to swap to potentially enable future removals or just find a better local optimum
        # Actually, if we can't remove, can we swap to a position that allows removal?
        # Or just try to find a smaller set via "k-exchange"?
        # For now, let's try 1-swap: replace c with c' if it helps? 
        # It doesn't help size directly.
        
        # Alternative: Randomized Restart
        # If stuck, restart greedy with random tie-breaking.
        
    # Convert indices back to words
    result = [idx_to_word(i) for i in current_code]
    
    # Final check: ensure distinct
    # The greedy adds distinct indices.
    
    return result

# EVOLVE-BLOCK-END
