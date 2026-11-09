"""Greedy set-cover with Local Search for ternary covering codes K_3(8,1)."""
import numpy as np
import random


def construct(q, n, radius):
    # For this task: q=3, n=8, radius=1
    N = q ** n
    
    # Precompute powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Precompute balls (list of indices at distance <= radius)
    # ball[i] = [i] + neighbors
    balls = []
    ball_sets = []
    for i in range(N):
        neighbors = [i]
        for p in range(n):
            cur = (i // powers[p]) % q
            base = i - cur * powers[p]
            for nd in range(q):
                if nd != cur:
                    nb = base + nd * powers[p]
                    neighbors.append(nb)
        balls.append(np.array(neighbors, dtype=np.int32))
        ball_sets.append(set(neighbors))
    
    # 1. Greedy Construction
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Precompute coverage potential for greedy? 
    # Just use the standard greedy loop but optimized with numpy sums.
    # Actually, the initial greedy in the original code was O(N^2).
    # We can speed it up by maintaining a 'coverage' count of uncovered words?
    # No, standard greedy picks max new coverage.
    
    # To make greedy faster, we can just run it. It's the bottleneck for initial setup.
    # But N=6561 is small enough.
    
    while np.any(uncovered):
        best_cov = -1
        best_i = -1
        unc = uncovered
        for i in range(N):
            # Calculate coverage of i over uncovered words
            # cov = sum(unc[balls[i]])
            cov = np.sum(unc[balls[i]])
            if cov > best_cov:
                best_cov = cov
                best_i = i
            elif cov == best_cov and i < best_i:
                best_i = i
        
        if best_i < 0:
            break
        
        code.append(best_i)
        uncovered[balls[best_i]] = False
        
    # 2. Local Search
    # Initialize coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for c in code:
        counts[balls[c]] += 1
        
    # Helper to check if removal of c is valid
    def can_remove(c):
        # Need counts[j] > 1 for all j in balls[c]
        return np.min(counts[balls[c]]) > 1

    # Helper to check if swap c->w is valid
    def can_swap(c, w):
        # Need counts[j] > 1 for all j in balls[c] \ balls[w]
        set_w = ball_sets[w]
        for j in balls[c]:
            if j not in set_w:
                if counts[j] <= 1:
                    return False
        return True

    # Iterative improvement
    max_iterations = 20 # Safety cap
    for _ in range(max_iterations):
        improved = False
        
        # Phase 1: Try to remove codewords
        # Iterate until no more removals possible in this pass
        removed_any = True
        while removed_any:
            removed_any = False
            for i in range(len(code)):
                c = code[i]
                if can_remove(c):
                    # Remove c
                    counts[balls[c]] -= 1
                    code.pop(i)
                    improved = True
                    removed_any = True
                    break # Restart scan as counts changed
            
        # Phase 2: Try to swap codewords to enable removals
        if not improved:
            # Try swaps
            # To avoid getting stuck in a local minimum of "no swap works",
            # we can try all swaps.
            # Optimization: Only try swaps if we expect it to help.
            # A swap is "good" if it allows a subsequent removal.
            # But checking that is expensive.
            # Let's just find ANY valid swap.
            
            swapped = False
            for i in range(len(code)):
                c = code[i]
                # Try all possible w
                # We can randomize the order of w to avoid bias?
                # Or just iterate.
                for w in range(N):
                    if w == c:
                        continue
                    if can_swap(c, w):
                        # Perform swap
                        counts[balls[c]] -= 1
                        counts[balls[w]] += 1
                        code[i] = w
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break # Restart whole loop to try removals again
            
    # Convert indices back to words
    result = []
    for idx in code:
        word = []
        rem = idx
        for p in range(n):
            digit = rem % q
            word.append(digit)
            rem //= q
        word.reverse()
        result.append(word)
    
    return result
