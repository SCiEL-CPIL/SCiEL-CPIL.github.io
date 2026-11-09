import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(3-1) = 15
    
    t_start = time.time()
    deadline = 115.0 # Safety margin (limit is 120s)
    
    # Precompute all words as indices 0..2186
    # index = d1*q^(n-1) + ... + dn
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Helper to get index from tuple
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    balls = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
    
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball
        
    # Precompute "covers" for each candidate codeword: which words it covers
    # This is the same as balls, but let's keep the naming clear.
    
    # --- Strategy: Algebraic Construction + Local Search ---
    
    # 1. Generate a good initial code using a structured approach.
    # The ternary Hamming code H_3(7) has parameters [7, 3, 3].
    # It is a perfect code for radius 1? No, perfect codes exist for radius 1 only if (q^n-1)/(q-1) is a power of q.
    # For q=3, n=7: (3^7-1)/2 = 1093, not a power of 3. So no perfect code.
    # However, the Hamming code structure provides a good starting point or a way to generate near-optimal codes.
    # Actually, a simpler and very effective strategy for small n is to use a greedy algorithm 
    # but initialize it with a "randomized greedy" or "weighted greedy" that has been pre-tuned.
    
    # Let's try a different angle: 
    # The lower bound is 156. The best known is 186.
    # A pure greedy usually lands around 200-220.
    # The LP relaxation gave ~186-190 in the seed.
    
    # New Structural Idea: 
    # Instead of solving a massive LP, use a **Greedy Set Cover with Randomized Restarts and Intensive Local Search**.
    # The local search will be a 1-exchange (remove one, add one) and 2-exchange (remove two, add two) 
    # to escape local minima. 
    # To make this fast, we precompute the "gain" of adding a word and "loss" of removing a word.
    
    # Let's implement a fast local search routine.
    
    def local_search(code_set, max_iters=1000):
        """
        Perform 1-exchange local search.
        code_set: set of indices
        Returns: improved set or None
        """
        code_list = list(code_set)
        covered = np.zeros(N, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        
        for idx in code_list:
            counts[balls[idx]] += 1
            covered[balls[idx]] = True
            
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try removing each codeword
            # If removing it doesn't uncover any words, remove it.
            for i in range(len(code_list)):
                idx = code_list[i]
                ball = balls[idx]
                # If min count in ball > 1, we can remove it
                if np.min(counts[ball]) > 1:
                    counts[ball] -= 1
                    code_list.pop(i)
                    code_set.discard(idx)
                    improved = True
                    break # Restart loop to avoid index issues
            
            if not improved:
                # Try 1-exchange: remove one, add one
                # This is expensive to do exhaustively. 
                # We can sample some removals and try to find a better addition.
                if random.random() < 0.5: # Do this with some probability to save time
                    # Pick a random codeword to remove
                    if code_list:
                        rem_idx = random.choice(code_list)
                        rem_ball = balls[rem_idx]
                        
                        # Temporarily remove
                        counts[rem_ball] -= 1
                        uncovered = np.zeros(N, dtype=bool)
                        for j in rem_ball:
                            if counts[j] == 0:
                                uncovered[j] = True
                            
                        if not np.any(uncovered):
                            # Safe removal
                            code_list.remove(rem_idx)
                            code_set.discard(rem_idx)
                            improved = True
                        else:
                            # Need to add a codeword that covers some of these
                            # Find the codeword that covers the most of 'uncovered'
                            # This is a max coverage problem on the subset 'uncovered'
                            # We can approximate by checking candidates that intersect 'uncovered'
                            
                            # To be fast, we only check a subset of candidates or use a heuristic
                            # For now, let's just revert if we don't find a quick win.
                            # A full 1-exchange search is O(N^2 * ball_size) which might be too slow if done naively.
                            # Let's try a simpler heuristic: 
                            # If we have time, we can try to add the best candidate.
                            pass
                        # Revert if we didn't improve
                        if not improved:
                            counts[rem_ball] += 1
                            
        return code_set

    best_code_set = set()
    best_len = N
    
    # Phase 1: Multiple Greedy Restarts
    # Use a randomized greedy where we pick from the top K candidates randomly.
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - t_start > deadline - 20:
            break
            
        covered = np.zeros(N, dtype=bool)
        code = []
        
        while not covered.all():
            if time.time() - t_start > deadline - 10:
                break
                
            # Calculate gains
            # To speed this up, we can maintain a list of candidates.
            # But for N=2187, calculating gains for all is ~2187 * 15 ops = ~30k ops.
            # This is fast in NumPy.
            
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(idx)
            covered[balls[idx]] = True
            
        # Local Search
        code_set = set(code)
        code_set = local_search(code_set, max_iters=50)
        
        if len(code_set) < best_len:
            best_len = len(code_set)
            best_code_set = code_set.copy()
            
    # Phase 2: Intensive Local Search on the best found so far
    # We can try to improve the best_code_set further with more iterations or different strategies.
    
    if best_code_set:
        # Try a few more rounds of local search with higher max_iters
        for _ in range(5):
            if time.time() - t_start > deadline - 5:
                break
            best_code_set = local_search(best_code_set, max_iters=200)
            
    # Fallback if nothing found (should not happen)
    if not best_code_set:
        covered = np.zeros(N, dtype=bool)
        best_code_set = set()
        while not covered.all():
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[0]
            best_code_set.add(idx)
            covered[balls[idx]] = True

    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code_set:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
