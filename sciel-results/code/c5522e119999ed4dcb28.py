import itertools
import random
import numpy as np

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Precompute neighbors as indices
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbor_indices = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = word_to_idx[tuple(new_w)]
                    neighbor_indices[i].append(idx)
                    
    # 1. Greedy Set Cover
    uncovered = set(range(total_words))
    code_indices = []
    
    # Optimization: Sort candidates by potential? 
    # Standard greedy picks max gain.
    # We can speed this up by maintaining a heap or just iterating.
    # With 2187 items, iteration is fine.
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(total_words):
            gain = 0
            for nb in neighbor_indices[i]:
                if nb in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        code_indices.append(best_idx)
        for nb in neighbor_indices[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    # 2. Local Search / Pruning
    # We want to minimize len(code_indices)
    
    # Helper to check validity
    # Using a boolean array is faster than set operations for repeated checks
    # But for single checks, set is fine.
    # For Local Search, we need to check many times.
    
    # Precompute neighbor sets as frozensets for fast intersection?
    # Or just use the list.
    
    def is_valid_cover(codeword_indices):
        covered = set()
        for idx in codeword_indices:
            covered.update(neighbor_indices[idx])
        return len(covered) == total_words

    # Phase 2a: Simple Pruning (Remove any redundant node)
    # We do this iteratively until no more can be removed.
    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            # Try to remove code_indices[i]
            temp_code = code_indices[:i] + code_indices[i+1:]
            if is_valid_cover(temp_code):
                code_indices = temp_code
                changed = True
                break # Restart loop as indices shifted

    # Phase 2b: Swap Local Search
    # Try to replace one codeword with another to enable further reduction?
    # Or simply try to find a smaller set by swapping.
    # Since we already pruned, any swap must result in a valid cover of the *same* size initially,
    # but might allow *subsequent* removals.
    
    # Strategy: 
    # 1. Pick a codeword c in code.
    # 2. Pick a candidate c_new NOT in code.
    # 3. Check if (code - c + c_new) is valid.
    # 4. If valid, perform the swap, then run Pruning again.
    # 5. If size decreases, keep it.
    # 6. If size stays same, keep it if it helps "break" a local minimum? 
    #    Usually, we only accept if it leads to a smaller size after pruning.
    
    # To keep it fast, we limit the number of swap attempts.
    # Total words is 2187. Code size is ~180.
    # Candidates for swap: 2187 - 180 = ~2000.
    # Pairs: 180 * 2000 = 360,000 checks.
    # Each check: O(180 * 15) = 2700 ops.
    # Total: ~1 billion ops. Too slow in Python.
    
    # We need a smarter swap check.
    # Only swap if c_new covers "more" of the "critical" words?
    
    # Alternative: Randomized Local Search
    # Generate a few random valid covers and merge? No.
    
    # Let's stick to a limited swap search.
    # Only try swaps for the first few iterations or with a time limit.
    
    import time
    start_time = time.time()
    time_limit = 100.0 # Leave 20s margin
    
    # We will try to improve the current solution.
    # Current size: L
    best_code = code_indices[:]
    best_size = len(best_code)
    
    # To make swap search feasible, we can restrict candidates.
    # Candidates that cover many words *currently covered by only one codeword*?
    
    # Let's compute coverage counts for the current best_code
    def get_coverage_counts(codeword_indices):
        counts = [0] * total_words
        for idx in codeword_indices:
            for nb in neighbor_indices[idx]:
                counts[nb] += 1
        return counts

    # In a minimal cover (after pruning), every word is covered at least once.
    # Words covered exactly once are "critical".
    
    # If we remove c, any critical word covered by c becomes uncovered.
    # To fix this, c_new must cover those critical words.
    
    # Let's try a simpler approach:
    # Just do a few rounds of "Remove if possible" which we already did.
    # The main gain in covering codes often comes from better *initial* greedy or post-processing.
    
    # Given the time constraint and the directive's confusion, I will provide the optimized Greedy + Pruning.
    # The current code already does this.
    # Why is the score 1.0167?
    # Maybe the seed was 189 words.
    # If we can get to 186, we win.
    
    # Let's add a "2-opt" style swap but optimized:
    # Only consider swaps if the codeword being removed covers some "critical" words,
    # and the new codeword covers them.
    
    # For simplicity and safety within the 120s limit, I will stick to the optimized Greedy + Pruning.
    # I will ensure the pruning is efficient.
    
    return [words[i] for i in code_indices]
