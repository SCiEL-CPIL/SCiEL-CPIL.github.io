import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # The task is specifically for q=3, n=6, radius=1
    # We will implement a robust, vectorized greedy set cover for this case.
    # For other parameters, we fall back to a simpler, potentially slower method or the same logic if dimensions are small.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Simple greedy with Python ints (slower but general)
        # This is rarely tested as the contract specifies the football pool problem.
        start_time = time.time()
        deadline = start_time + 110
        total_words = q ** n
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors as lists for general case
        neighbors = []
        for x in range(total_words):
            nb_set = {x}
            for i in range(n):
                digit = (x // powers[i]) % q
                for delta in range(1, radius + 1):
                    if digit - delta >= 0:
                        nb_set.add(x - delta * powers[i])
                    if digit + delta < q:
                        nb_set.add(x + delta * powers[i])
            neighbors.append(list(nb_set))
            
        all_mask = (1 << total_words) - 1
        best_code = None
        best_size = total_words
        
        for trial in range(50):
            if time.time() > deadline:
                break
            uncovered = all_mask
            code = []
            while uncovered:
                max_cov = -1
                best_cands = []
                for w in range(total_words):
                    # Count overlap using set intersection logic or bitmask
                    # For small total_words, set intersection is fine.
                    # But bitmask is faster.
                    # We don't have bitmasks precomputed for general case in this branch.
                    # Let's just use a simple count.
                    # This branch is a fallback, so performance is less critical than correctness.
                    cov = 0
                    for nb in neighbors[w]:
                        if (uncovered >> nb) & 1:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_cands = [w]
                    elif cov == max_cov:
                        best_cands.append(w)
                if not best_cands:
                    break
                chosen = random.choice(best_cands)
                code.append(chosen)
                for nb in neighbors[chosen]:
                    uncovered &= ~(1 << nb)
                    
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
        if best_code:
            res = []
            for idx in best_code:
                word = []
                temp = idx
                for i in range(n):
                    word.append(temp % q)
                    temp //= q
                res.append(word)
            return res
        else:
            return [[0]*n]

    # Optimized path for q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for evaluation overhead
    
    N = 729
    powers = [1, 3, 9, 27, 81, 243]
    
    # Precompute the coverage matrix C (729x729) as a numpy array of uint8
    # C[w, u] = 1 if u is in the ball of w
    # This is a constant cost precomputation.
    C = np.zeros((N, N), dtype=np.uint8)
    for w in range(N):
        # w itself
        C[w, w] = 1
        for i in range(6):
            digit = (w // powers[i]) % 3
            if digit != 0:
                C[w, w - powers[i]] = 1
            if digit != 2:
                C[w, w + powers[i]] = 1
                
    # We want to find a set of rows that cover all columns.
    # Greedy:
    # 1. Initialize uncovered = ones(N)
    # 2. While uncovered.any():
    #    a. Compute counts = C @ uncovered
    #    b. Pick w = argmax(counts)
    #    c. code.append(w)
    #    d. uncovered = uncovered & ~C[w]
    
    best_code = None
    best_size = N + 1
    
    # Number of trials
    # Each trial is fast. Let's estimate.
    # Matrix vector product 729x729 * 729x1. ~500k ops.
    # 73 steps per trial. ~36M ops per trial.
    # NumPy can do ~100M-1G ops/s. So ~0.1-1s per trial?
    # Actually, BLAS is very fast. Let's say 0.05s per trial.
    # 115s / 0.05s = 2300 trials.
    # We can run many trials.
    
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        while uncovered.any():
            # Compute marginal gain for each candidate
            # C is uint8, uncovered is uint8.
            # Result will be int (sum of products).
            # Using np.dot is efficient.
            counts = C.dot(uncovered)
            
            # Find the candidate with max count
            # If multiple, we can pick one randomly or the first.
            # Randomization helps escape local minima in the sequence of choices.
            max_count = np.max(counts)
            
            # Get all indices with max count
            # To avoid overhead of np.where if there's only one, we can just take argmax.
            # But random tie-breaking is better.
            # Let's just take argmax for speed, or random from ties if we want.
            # np.where(np.max(counts) == counts)[0]
            # Let's stick to argmax for speed, the randomness comes from the fact that
            # if we had a tie, argmax picks the first one. 
            # To add randomness, we can shuffle the candidates or add small noise?
            # Better: if there are ties, pick randomly.
            # But checking ties is O(N).
            # Let's just use argmax. The greedy choice is deterministic given the state.
            # To diversify, we can randomize the order of evaluation? No, C is fixed.
            # We can randomize the initial state? No, uncovered is always full.
            # So pure greedy is deterministic!
            # We need to introduce randomness.
            # Option 1: Randomly break ties.
            # Option 2: Randomly perturb the matrix C? No, must be valid.
            # Option 3: Randomly select from top K candidates?
            
            # Let's get the top candidates.
            # For speed, just argmax.
            # To make it stochastic, we can shuffle the columns of C? No.
            # We can shuffle the rows? No.
            
            # Actually, standard greedy is deterministic.
            # To get different solutions, we can use "Randomized Greedy":
            # Instead of picking the absolute max, pick from the set of candidates
            # with count >= max_count * (1 - epsilon)?
            # Or simply: if there are ties, pick randomly.
            
            # Let's check for ties efficiently.
            # max_val = np.max(counts)
            # ties = np.where(counts == max_val)[0]
            # if len(ties) > 1: chosen = np.random.choice(ties)
            # else: chosen = ties[0]
            
            # This adds O(N) overhead per step. 729 * 73 * 5000 trials = ~2.6B ops? No, 729*73 is ~50k per trial. 50k * 5000 = 250M ops. This is fine.
            
            max_val = np.max(counts)
            # Find indices where counts == max_val
            # Using np.where is efficient.
            candidates = np.where(counts == max_val)[0]
            
            if len(candidates) == 0:
                break # Should not happen if uncovered is not empty and C is valid
                
            chosen = np.random.choice(candidates)
            
            code.append(chosen)
            
            # Update uncovered
            # uncovered = uncovered & ~C[chosen]
            # C[chosen] is a row of 0/1s.
            # ~C[chosen] is 1/0s (inverted).
            # Since they are uint8, ~ might behave oddly with bitwise not on uint8.
            # ~0 = -1 (all bits 1), ~1 = -2 (all bits 1 except LSB).
            # So we should use: uncovered = uncovered * (1 - C[chosen])
            # Or: uncovered[np.where(C[chosen] == 1)[0]] = 0
            
            # Vectorized update:
            uncovered &= (1 - C[chosen])
            
        # End while uncovered
        
        # Reduction Phase:
        # Try to remove redundant codewords.
        # A codeword c is redundant if the union of balls of (code \ {c}) covers the space.
        # This is equivalent to: for all u in Space, exists c' in code\{c} such that u in Ball(c').
        
        if len(code) < best_size:
            best_size = len(code)
            # Deep copy
            best_code = code.copy()
            
        # Perform reduction
        # We iterate through the code and try to remove each.
        # To be efficient, we can check redundancy by summing the coverage of the others.
        # If sum(C[c'] for c' in others) >= 1 for all u, then c is redundant.
        
        # Let's do a single pass reduction.
        # We can re-run greedy on the remaining points if we remove one? No, that changes the problem.
        # Just check if removal is safe.
        
        # Optimization: Only check removal if the code size is small enough or we have time.
        # With ~73 codewords, checking each one requires summing 72 vectors of length 729.
        # 72 * 729 ~ 50k ops per check. 73 checks ~ 3.6M ops.
        # This is fast.
        
        removed_any = True
        while removed_any and time.time() < deadline - 2:
            removed_any = False
            current_code = code[:]
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                # Check if c is redundant
                # Calculate coverage of all other codewords
                # We can accumulate a boolean mask
                covered_by_others = np.zeros(N, dtype=np.uint8)
                for j in range(len(current_code)):
                    if j == i:
                        continue
                    covered_by_others |= C[current_code[j]]
                    
                if not covered_by_others.any():
                    # If no others, and N>0, it's not redundant (unless N=0)
                    # But if N>0 and only c is in code, removing c leaves uncovered.
                    pass
                else:
                    # Check if covered_by_others covers everything
                    # If sum(covered_by_others) == N, then c is redundant
                    if np.sum(covered_by_others) == N:
                        # Remove c
                        code.pop(i)
                        removed_any = True
                        break # Restart the while loop to check again
                        # Actually, we can just continue checking others, but breaking is safer for correctness
                        # and allows re-evaluation.
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()

    if best_code:
        res = []
        for idx in best_code:
            word = []
            temp = int(idx)
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        return res
    else:
        # Fallback: Return a known valid code (e.g. from a simple construction)
        # The tetracode construction mentioned in the seed:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
