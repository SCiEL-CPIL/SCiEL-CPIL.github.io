import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters for the task
    # q=3, n=6, radius=1
    total_words = q ** n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit

    # Generate all possible words as indices 0 to q^n - 1
    # Convert index to base-q representation
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage masks for all possible codewords
    # masks[c] is a boolean array of length total_words, True if word c covers word w
    masks = np.zeros((total_words, total_words), dtype=bool)
    
    for c_idx in range(total_words):
        c_word = idx_to_word(c_idx)
        masks[c_idx, c_idx] = True # Distance 0
        
        # Distance 1 neighbors
        for pos in range(n):
            orig_val = c_word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    neighbor = c_word.copy()
                    neighbor[pos] = new_val
                    n_idx = word_to_idx(neighbor)
                    masks[c_idx, n_idx] = True

    def greedy_random():
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        cover_counts = np.zeros(total_words, dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Calculate coverage for all candidates on currently uncovered words
            # masks shape: (total_words, total_words)
            # uncovered shape: (total_words,)
            # We want sum over axis=1 of (masks & uncovered[None, :])
            
            # Optimization: Only compute for candidates not yet selected?
            # Actually, selected candidates have 0 new coverage because we update uncovered.
            # So we can just compute for all, or mask out selected.
            # Computing for all is simpler and NumPy is fast.
            
            coverage_counts = np.sum(masks & uncovered[None, :], axis=1)
            
            max_cov = np.max(coverage_counts)
            if max_cov == 0:
                break
                
            # Find all candidates with max coverage
            candidates = np.where(coverage_counts == max_cov)[0]
            
            # Randomly pick one to break symmetry
            # Optional: Symmetry break for the first selection
            if not selected:
                # All candidates are symmetric. Pick 0 to canonicalize.
                # But 0 might not be in candidates if we started with random?
                # No, initially all cover same.
                # We can just pick candidates[0] or random.
                # Picking 0 is a good symmetry break.
                if 0 in candidates:
                    chosen = 0
                else:
                    chosen = random.choice(candidates)
            else:
                chosen = random.choice(candidates)
            
            selected.append(chosen)
            # Update uncovered
            uncovered[masks[chosen]] = False
            # Update cover counts for deletion phase
            cover_counts[masks[chosen]] += 1

        return selected, cover_counts

    def minimize_code(selected, cover_counts):
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                # Try removing c
                cover_counts[masks[c]] -= 1
                if np.all(cover_counts > 0):
                    # Safe to remove
                    selected.pop(i)
                    changed = True
                else:
                    # Restore
                    cover_counts[masks[c]] += 1

        return selected

    best_size = float('inf')
    best_selected = None
    
    # Run multiple restarts
    # Estimate time per run: ~0.2s greedy + ~0.01s minimize
    # We can do ~100 runs in 100s?
    # Let's just run until deadline.
    
    while time.time() < deadline:
        selected, counts = greedy_random()
        
        if time.time() > deadline:
            break
            
        selected = minimize_code(selected, counts)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
        # If we reached the known best, we can stop early?
        # The task says best known is 73.
        # If we get 73, we are at the limit of known possibilities (likely).
        # But we can try to see if we can get 72? Lower bound is 71.
        # So we shouldn't stop at 73, but 73 is a very good score (1.0).
        # If we get 73, we can probably stop to save time?
        # No, if we get 73, score is 1.0. If we get 72, score is 73/72 > 1.0.
        # So keep going.
        
        # Heuristic: If we get stuck at 74 for many runs, maybe stop?
        # No, just keep trying.

    if best_selected is None:
        # Fallback: return a valid code if no loop completed?
        # Greedy always completes.
        selected, counts = greedy_random()
        selected = minimize_code(selected, counts)
        best_selected = selected

    return [idx_to_word(idx) for idx in best_selected]
