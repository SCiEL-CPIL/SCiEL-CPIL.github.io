# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius r.
    Uses vectorized greedy + local search with critical-word analysis and swap moves.
    """
    t0 = time.time()
    deadline = t0 + 120.0  # hard safety margin (evaluator kills at 150s)
    
    # For the specific task: q=3, n=7, radius=1
    # Total space size
    space_size = q ** n  # 2187 for the task
    
    # Precompute all words as integers 0..space_size-1
    all_words = np.arange(space_size, dtype=np.int32)
    
    # Convert word index to digit vector (for distance computation)
    # We'll work with digit vectors for clarity
    
    # Helper: get digits of an integer
    def int_to_digits(x, q, n):
        digits = np.empty(n, dtype=np.int32)
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    # Precompute digit vectors for all words (n x space_size matrix)
    digits_all = np.empty((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for j in range(n):
            digits_all[i, j] = x % q
            x //= q
    
    # Precompute distance between any word and any codeword candidate
    # dist_matrix[i, c] = hamming distance between word i and codeword c
    # For greedy: for each word, find if it's covered
    
    # Strategy:
    # 1. Greedy: repeatedly pick the word that covers the most uncovered words
    # 2. Local search: try swaps (remove one, add one) to reduce count
    # 3. Multi-restart with different random seeds for diversification
    
    def ball_mask(cw_idx):
        """Return boolean mask of words within radius of codeword cw_idx."""
        # Hamming distance between cw_idx and all words
        diff = np.abs(digits_all[cw_idx] - digits_all)  # (space_size, n)
        dist = np.sum(diff != 0, axis=1)  # (space_size,)
        return dist <= radius
    
    def is_covered(codewords, all_words_idx):
        """Check if all words are within radius of some codeword."""
        if len(codewords) == 0:
            return False
        covered = np.zeros(space_size, dtype=bool)
        for cw in codewords:
            covered |= ball_mask(cw)
        return np.all(covered)
    
    def count_uncovered(codewords, all_words_idx):
        """Return number of uncovered words."""
        if len(codewords) == 0:
            return space_size
        covered = np.zeros(space_size, dtype=bool)
        for cw in codewords:
            covered |= ball_mask(cw)
        return np.sum(~covered)
    
    # Greedy construction
    def greedy_construction(seed=None):
        rng = np.random.RandomState(seed)
        uncovered = np.ones(space_size, dtype=bool)
        codewords = []
        
        while np.any(uncovered):
            # Find the word that covers the most uncovered words
            best_cw = -1
            best_count = -1
            # Sample candidates for speed (full scan is expensive)
            candidates = np.where(uncovered)[0]
            if len(candidates) > 200:
                candidates = candidates[rng.choice(len(candidates), size=200, replace=False)]
            for cw in candidates:
                mask = ball_mask(cw)
                count = np.sum(uncovered & mask)
                if count > best_count:
                    best_count = count
                    best_cw = cw
            if best_cw < 0:
                break
            codewords.append(best_cw)
            uncovered &= ~ball_mask(best_cw)
        
        return codewords
    
    # Local search: try to remove a codeword if coverage is maintained, or swap
    def local_search(codewords, max_iter=10000, seed=None):
        rng = np.random.RandomState(seed)
        codewords = list(codewords)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Try removing each codeword
            for i in range(len(codewords)):
                if time.time() > deadline - 5:
                    return codewords
                trial = codewords[:i] + codewords[i+1:]
                if is_covered(trial, all_words):
                    codewords = trial
                    improved = True
                    break
            # Try swaps: remove one, add one
            if not improved and len(codewords) > 0:
                for i in range(len(codewords)):
                    if time.time() > deadline - 5:
                        return codewords
                    trial = codewords[:i] + codewords[i+1:]
                    # Find an uncovered word if any, and add a codeword covering it
                    covered = np.zeros(space_size, dtype=bool)
                    for cw in trial:
                        covered |= ball_mask(cw)
                    uncovered_idx = np.where(~covered)[0]
                    if len(uncovered_idx) > 0:
                        # Pick a random uncovered word and find best replacement
                        uw = uncovered_idx[rng.randint(len(uncovered_idx))]
                        # Find codeword that covers most of the uncovered + uw
                        best_new = -1
                        best_gain = -1
                        for cw in all_words:
                            mask = ball_mask(cw)
                            gain = np.sum(~covered & mask)
                            if gain > best_gain:
                                best_gain = gain
                                best_new = cw
                        if best_new >= 0 and best_new not in trial:
                            trial = trial + [best_new]
                            if is_covered(trial, all_words) and len(trial) < len(codewords):
                                codewords = trial
                                improved = True
                                break
        return codewords
    
    # Multi-restart greedy + local search
    best_solution = None
    best_size = float('inf')
    
    # Initial greedy
    init = greedy_construction(seed=42)
    init = local_search(init, max_iter=5000, seed=42)
    best_solution = init
    best_size = len(init)
    
    # Multi-restart
    restart_seed = 0
    while time.time() < deadline - 10:
        restart_seed += 1
        sol = greedy_construction(seed=restart_seed)
        sol = local_search(sol, max_iter=3000, seed=restart_seed)
        if len(sol) < best_size:
            best_solution = sol
            best_size = len(sol)
            # Early exit if we hit a very good solution
            if best_size <= 170:
                break
    
    # Final polish: try more aggressive local search on the best solution
    if time.time() < deadline - 5:
        best_solution = local_search(best_solution, max_iter=20000, seed=999)
    
    # Convert integer indices to digit lists
    result = []
    for cw_idx in best_solution:
        result.append(digits_all[cw_idx].tolist())
    
    # Ensure distinctness
    seen = set()
    final = []
    for r in result:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            final.append(r)
    
    return final
# EVOLVE-BLOCK-END
