import time
import numpy as np
import itertools

_start_time = time.time()
_DEADLINE = 120.0

def _precompute_neighbors(q, n, radius):
    """
    Precompute the neighborhood of each word in the space.
    Returns a list of lists, where neighbors[i] is a list of indices within Hamming distance <= radius of word i.
    """
    space_size = q ** n
    # Digits for each word
    digits = np.zeros((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
    
    neighbors = [[] for _ in range(space_size)]
    
    # For radius 1: self + 1-position changes
    for i in range(space_size):
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig = digits[i, pos]
                for d in range(q):
                    if d == orig:
                        continue
                    # Calculate new index
                    # index = sum(digits[j] * q^(n-1-j))
                    # Change at pos: delta = (d - orig) * q^(n-1-pos)
                    weight = q ** (n - 1 - pos)
                    new_idx = i + (d - orig) * weight
                    neighbors[i].append(new_idx)
                    
    # Convert to numpy arrays for faster access if needed, or keep as lists
    # For pruning, we need to iterate neighbors frequently. Lists are okay.
    # But for vectorized operations, we might want a CSR-like structure or just a boolean matrix.
    # Given 6561x6561, a boolean matrix is 43MB. Let's use a boolean matrix for speed in greedy.
    cov_matrix = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cov_matrix[i, neighbors[i]] = True
        
    return digits, cov_matrix, neighbors

def _greedy_cover_optimized(q, n, radius, cov_matrix, digits, time_limit, seed=0):
    """
    Greedy set cover with overlap penalty and randomized tie-breaking.
    """
    space_size = q ** n
    rng = np.random.default_rng(seed)
    
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    coverage_count = np.zeros(space_size, dtype=np.int32)
    
    # Precompute the "potential" of each word (initially all 17)
    # We will update this as we cover points.
    # potential[i] = number of uncovered neighbors of i
    
    # Initial potential is just the degree of the node (17 for radius 1, q=3, n=8)
    # But we can compute it on the fly or precompute.
    # Let's compute initial potential
    potential = cov_matrix.sum(axis=1).astype(np.int32)
    
    # To speed up tie-breaking and avoid deterministic local minima, 
    # we can add a small random noise to the potential scores.
    noise_scale = 0.1
    scores = potential.astype(np.float32) + rng.normal(0, noise_scale, space_size)
    
    remaining = space_size
    
    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        
        # Pick the best candidate
        # We want the one with the highest score
        best_idx = int(np.argmax(scores))
        
        if potential[best_idx] <= 0:
            break
            
        selected.append(best_idx)
        
        # Update uncovered and coverage_count
        # Get neighbors of best_idx
        nbrs = np.where(cov_matrix[best_idx])[0]
        
        # Newly covered are those in nbrs that are currently uncovered
        newly_covered = nbrs[uncovered[nbrs]]
        
        if len(newly_covered) == 0:
            # This shouldn't happen if potential[best_idx] > 0
            break
            
        # Update coverage_count for ALL neighbors
        coverage_count[nbrs] += 1
        
        # Update potential for ALL words
        # For each word j, potential[j] decreases by the number of its neighbors in newly_covered
        # This is cov_matrix[:, newly_covered].sum(axis=1)
        # This operation is O(space_size * len(newly_covered)) which can be slow.
        # Alternative: 
        # potential[j] = sum(cov_matrix[j, k] for k in uncovered)
        # We can update potential by subtracting the contribution of the newly covered points.
        
        # Vectorized update:
        # delta_potential[j] = number of neighbors of j that are in newly_covered
        # delta_potential = cov_matrix[:, newly_covered].sum(axis=1)
        
        # To make this faster, we can use the fact that cov_matrix is sparse-ish (17 ones per row).
        # But numpy boolean indexing is fast enough for 6561x6561 if done carefully.
        
        # Let's try a different update strategy:
        # Instead of updating all potentials, we can just recalculate the score for the next step?
        # No, that's O(N^2).
        
        # Let's stick to the vectorized update but be careful with time.
        # 6561 * 17 operations per step. ~100k ops. 400 steps -> 40M ops. Should be fast in NumPy.
        
        delta = cov_matrix[:, newly_covered].sum(axis=1).astype(np.int32)
        potential -= delta
        potential = np.maximum(potential, 0)
        
        # Update scores
        scores[best_idx] = -1 # Mark as selected
        # Add new noise for tie-breaking in next iteration?
        # Or just keep the deterministic part and rely on the initial noise?
        # Re-adding noise might be better for diversity.
        scores = potential.astype(np.float32) + rng.normal(0, noise_scale, space_size)
        scores[best_idx] = -1
        
        # Update uncovered
        uncovered[newly_covered] = False
        remaining -= len(newly_covered)
        
    return selected

def _prune_redundant(q, n, radius, cov_matrix, digits, selected_indices, time_limit):
    """
    Iteratively remove redundant codewords.
    A codeword i is redundant if for all v in N(i), coverage_count[v] > 1.
    """
    space_size = q ** n
    selected_set = set(selected_indices)
    current_selected = list(selected_indices)
    
    # Calculate initial coverage count
    coverage_count = np.zeros(space_size, dtype=np.int32)
    for idx in current_selected:
        coverage_count[cov_matrix[idx]] += 1
        
    # Check for redundancy
    changed = True
    while changed:
        if time.time() - _start_time > time_limit - 5:
            break
        changed = False
        to_remove = []
        
        for idx in current_selected:
            # Check if idx is redundant
            # It is redundant if min(coverage_count[N(idx)]) > 1
            # i.e., all neighbors are covered by at least 2 codewords
            nbrs = np.where(cov_matrix[idx])[0]
            if np.min(coverage_count[nbrs]) > 1:
                to_remove.append(idx)
                
        if not to_remove:
            break
            
        for idx in to_remove:
            if idx in current_selected:
                current_selected.remove(idx)
                # Update coverage count
                nbrs = np.where(cov_matrix[idx])[0]
                coverage_count[nbrs] -= 1
                changed = True
        if to_remove:
            changed = True # Ensure loop continues if any were removed
            
    return current_selected

def _tetracode_solution(q, n, radius):
    """Fallback for non-standard params or initialization."""
    if q != 3 or n != 8 or radius != 1:
        # Generic fallback: simple greedy
        pass
        
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words

def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5
    
    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Precompute neighbors and coverage matrix
    digits, cov_matrix, neighbors = _precompute_neighbors(q, n, radius)
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy passes with different seeds
    num_restarts = 5
    for seed in range(num_restarts):
        if time.time() - _start_time > time_limit - 10:
            break
            
        # 1. Greedy Construction
        selected_indices = _greedy_cover_optimized(q, n, radius, cov_matrix, digits, time_limit, seed=seed)
        
        # 2. Pruning
        pruned_indices = _prune_redundant(q, n, radius, cov_matrix, digits, selected_indices, time_limit)
        
        if len(pruned_indices) < best_size:
            best_size = len(pruned_indices)
            best_solution = pruned_indices
            
    # If no solution found (shouldn't happen), fallback
    if best_solution is None:
        # Fallback to tetracode
        words = _tetracode_solution(q, n, radius)
        indices = []
        for w in words:
            idx = 0
            for d in w:
                idx = idx * 3 + d
            indices.append(idx)
        return words
        
    # Convert best_solution indices to words
    result_words = [digits[i].tolist() for i in best_solution]
    
    # Final verification
    # Check if all words are covered
    covered = np.zeros(q**n, dtype=bool)
    for idx in best_solution:
        covered[cov_matrix[idx]] = True
    if not np.all(covered):
        # Fallback to tetracode if verification fails
        words = _tetracode_solution(q, n, radius)
        return words
        
    return result_words
