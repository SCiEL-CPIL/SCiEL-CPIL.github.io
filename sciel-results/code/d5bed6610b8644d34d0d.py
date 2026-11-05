"""Optimized Greedy Set-Cover with NumPy vectorization and strict time guards."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _id_to_word(wid, q, n):
    """Convert integer id to list of digits."""
    d = [0] * n
    x = wid
    for p in range(n):
        d[p] = x % q
        x //= q
    return d


def _word_to_id(word, q, n):
    """Convert list of digits to integer id."""
    x = 0
    for p in range(n):
        x = x * q + word[p]
    return x


def _generate_all_words(q, n):
    """Generate all words as a NumPy array of shape (q^n, n)."""
    total = q ** n
    words = np.empty((total, n), dtype=np.int8)
    for i in range(total):
        x = i
        for p in range(n):
            words[i, p] = x % q
            x //= q
    return words


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin (limit is 120s, allow 10s buffer)
    total = q ** n
    
    # Precompute all words as a matrix for vectorized distance calculation
    # Shape: (total, n)
    all_words = _generate_all_words(q, n)
    
    # We want to cover all words (0..total-1)
    # A codeword c covers a word w if dist(c, w) <= radius
    
    # Precompute the "coverage mask" for each codeword?
    # total x total boolean matrix might be 729x729 = 531k booleans. This is small (~500KB).
    # Let's precompute the coverage matrix once. This is much faster than recalculating distances in each greedy step.
    
    # Calculate pairwise Hamming distances
    # all_words is (N, n). We need dist(i, j) for all i, j.
    # dist(i, j) = sum(all_words[i] != all_words[j])
    # Using broadcasting: (N, 1, n) vs (1, N, n) -> (N, N, n)
    # This creates an array of size N*N*n. For N=729, n=6: 729*729*6 ~ 3.2M elements.
    # This is very fast in NumPy.
    
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dists = np.sum(diff, axis=2)
    
    # Coverage matrix: cover[i, j] is True if word i covers word j (dist <= radius)
    # Note: In set cover, we pick codewords (columns) to cover targets (rows).
    # Let's say rows are targets (all words), columns are candidates (all words).
    # cover[target, candidate] = (dist(candidate, target) <= radius)
    # Since dist is symmetric, cover is symmetric.
    cover_matrix = dists <= radius
    
    # We will run greedy set cover multiple times with different random tie-breakers.
    # State: uncovered_mask (boolean array of length total)
    # Initial: all True.
    
    best_solution = None
    best_len = total + 1
    
    # Run multiple random restarts
    rng = random.Random(42)
    
    # To speed up greedy selection, we maintain an array of "gains" for each candidate.
    # gain[c] = number of currently uncovered targets covered by candidate c.
    
    # Precompute for each candidate c, the list of targets it covers?
    # Or just use the matrix.
    
    # Efficient Greedy:
    # 1. Start with uncovered = np.ones(total, dtype=bool)
    # 2. While uncovered.any():
    #    a. Compute gains: gains = cover_matrix[:, uncovered].sum(axis=1) 
    #       Wait, cover_matrix is (target, candidate).
    #       We want for each candidate c, how many *uncovered* targets it covers.
    #       So we need sum over targets where uncovered is True.
    #       gains = np.sum(cover_matrix[:, uncovered], axis=0) # Shape (total,)
    #    b. Find max gain.
    #    c. Tie-break randomly among max gains.
    #    d. Add to solution.
    #    e. Update uncovered: uncovered[cover_matrix[:, chosen]] = False
    
    for restart in range(50): # Fixed number of restarts, time guard will stop early if needed
        if time.time() > deadline:
            break
            
        uncovered = np.ones(total, dtype=bool)
        chosen = []
        
        # Optimization: Instead of slicing cover_matrix[:, uncovered] which creates a view or copy,
        # we can maintain a count array.
        # But with N=729, the vectorized operation is likely faster than Python loops.
        
        while uncovered.any():
            if time.time() > deadline:
                break
                
            # Calculate gains for all candidates
            # cover_matrix is (N, N). Rows are targets, cols are candidates.
            # We want sum of cover_matrix[i, c] for all i where uncovered[i] is True.
            # This is effectively: cover_matrix[:, :][uncovered, :].sum(axis=0)
            # Or simpler: (cover_matrix * uncovered[:, np.newaxis]).sum(axis=0)
            # Using boolean mask multiplication is fast.
            
            # Even faster:
            # gains = np.sum(cover_matrix[:, uncovered], axis=0) 
            # Note: cover_matrix[:, uncovered] selects columns corresponding to uncovered targets?
            # No, cover_matrix[:, k] is the column for candidate k.
            # We want to sum down the rows (targets) for each column (candidate), but only for rows that are uncovered.
            # So:
            masked_cover = cover_matrix[:, uncovered] # Shape (N_uncovered, N_candidates)
            gains = masked_cover.sum(axis=0) # Shape (N_candidates,)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if cover matrix is correct and uncovered is non-empty
                break
                
            # Find all candidates with max_gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-break
            chosen_id = candidates[rng.randint(0, len(candidates) - 1)]
            
            chosen.append(chosen_id)
            
            # Update uncovered
            # A target is covered if it is within radius of chosen_id.
            # This is cover_matrix[:, chosen_id]
            uncovered[cover_matrix[:, chosen_id]] = False
            
        if time.time() > deadline:
            break
            
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_solution = chosen[:]
            
            # If we reached the theoretical lower bound or a very good bound, we could stop?
            # Lower bound is 71. If we hit 71, we can stop.
            if best_len <= 71:
                break

    if best_solution is None:
        # Fallback: return a valid but potentially suboptimal solution
        # Just return all words (valid, but score is terrible)
        # Or a simple greedy run without restarts
        uncovered = np.ones(total, dtype=bool)
        chosen = []
        while uncovered.any():
            gains = np.sum(cover_matrix[:, uncovered], axis=0)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            chosen_id = candidates[0]
            chosen.append(chosen_id)
            uncovered[cover_matrix[:, chosen_id]] = False
        best_solution = chosen

    # Convert ids to words
    return [_id_to_word(wid, q, n) for wid in best_solution]
# EVOLVE-BLOCK-END
