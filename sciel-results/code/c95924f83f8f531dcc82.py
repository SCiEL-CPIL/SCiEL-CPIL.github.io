"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard cases
    if q != 3 or n != 6 or radius != 1:
        if n < 4:
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            while uncovered:
                best_word = None
                best_count = -1
                for w in all_words:
                    count = sum(1 for u in uncovered if sum(1 for a, b in zip(w, u) if a != b) <= radius)
                    if count > best_count:
                        best_count = count
                        best_word = w
                code.append(list(best_word))
                to_remove = set()
                for u in uncovered:
                    if sum(1 for a, b in zip(best_word, u) if a != b) <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            return code
        else:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1 using Simulated Annealing
    total_words = q ** n
    indices = list(range(total_words))
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighborhoods
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Convert back to idx
                idx_new = 0
                for d in range(n):
                    idx_new = idx_new * q + new_w[d]
                neigh.append(idx_new)
        neighborhoods.append(neigh)

    # Helper to calculate cost
    # Cost = num_codewords + penalty * num_uncovered
    penalty = 1000

    def calc_cost(code_set, covered_mask):
        num_uncovered = np.sum(~covered_mask)
        return len(code_set) + penalty * num_uncovered

    # 1. Initialize with Greedy Set Cover to get a good starting point
    uncovered = set(indices)
    initial_code = set()
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in indices:
            count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        initial_code.add(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.discard(nb)
        
    # Convert to numpy mask for SA
    current_code = set(initial_code)
    covered_mask = np.zeros(total_words, dtype=bool)
    for c in current_code:
        for nb in neighborhoods[c]:
            covered_mask[nb] = True
    
    current_cost = calc_cost(current_code, covered_mask)
    best_code = set(current_code)
    best_cost = current_cost
    
    # 2. Simulated Annealing
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < max_time:
        # Generate neighbor
        # 50% chance add, 50% chance remove
        if len(current_code) > 0 and random.random() < 0.5:
            # Remove
            to_remove = random.choice(list(current_code))
            new_code = current_code - {to_remove}
            # Update covered mask
            # We need to recompute covered mask for the new code
            # This is O(|code| * |neighborhood|)
            # Optimization: Only update if we track counts, but for simplicity and speed in Python,
            # recomputing 729 bools is fast.
            new_covered = np.zeros(total_words, dtype=bool)
            for c in new_code:
                new_covered[neighborhoods[c]] = True
        else:
            # Add
            # Pick a random word not in code
            # To avoid O(N) search for a word not in code, we can just pick random idx
            # and check. Or maintain a list of candidates.
            # Simple approach: pick random idx
            candidate = random.randint(0, total_words - 1)
            if candidate in current_code:
                continue # Skip this move
            new_code = current_code | {candidate}
            new_covered = np.zeros(total_words, dtype=bool)
            for c in new_code:
                new_covered[neighborhoods[c]] = True

        new_cost = calc_cost(new_code, new_covered)
        delta = new_cost - current_cost
        
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_code = new_code
            covered_mask = new_covered
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_code = set(current_code)
        
        T *= cooling

    # Return best valid code found
    # Verify best_code is valid
    final_covered = np.zeros(total_words, dtype=bool)
    for c in best_code:
        final_covered[neighborhoods[c]] = True
        
    if not np.all(final_covered):
        # Fallback to initial greedy code if somehow invalid (shouldn't happen)
        return [idx_to_word(c) for c in initial_code]
        
    return [idx_to_word(c) for c in best_code]

# EVOLVE-BLOCK-END
