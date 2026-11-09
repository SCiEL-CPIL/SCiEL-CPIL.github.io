"""Covering code search for K_q(n, radius) using algebraic symmetry + fast greedy set-cover."""
import time
import numpy as np
import random
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _build_coverage_matrix(q, n, radius):
    """
    Build a coverage matrix M of shape (total, total) where
    M[i, j] = 1 if word i covers word j (i.e., hamming(word_i, word_j) <= radius).
    """
    total = q ** n
    all_words = np.array([_idx_to_word(i, n, q) for i in range(total)], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    
    M = (dists <= radius).astype(np.int8)
    return M, all_words


def _direct_sum_tetracode(q, n, radius):
    """
    Direct sum of ternary tetracode (9 words of length 4) with all tails.
    This provides a valid covering code of size 9 * q^(n-4).
    For q=3, n=6, size is 9 * 3^2 = 81.
    """
    if q != 3 or n < 4:
        return None
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n == 4:
        return [list(h) for h in tetracode]
    
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]


def _generate_algebraic_candidates(q, n, radius):
    """
    Generate a candidate pool using algebraic symmetries:
    - Start with tetracode-based words
    - Apply cyclic shifts of coordinates
    - Apply all coordinate permutations
    - Apply value shifts (add constant mod q to all positions)
    - Apply value permutations (for q=3: swap 1 and 2)
    This creates a large but structured candidate set.
    """
    if q != 3 or n != 6:
        # Fallback: use tetracode + random
        tetracode_code = _direct_sum_tetracode(q, n, radius)
        if tetracode_code is not None:
            candidates = set()
            for w in tetracode_code:
                candidates.add(tuple(w))
            # Add some random candidates
            rng = random.Random(123)
            for _ in range(200):
                w = [rng.randint(0, q-1) for _ in range(n)]
                candidates.add(tuple(w))
            return list(candidates)
    
    # For q=3, n=6: use rich algebraic structure
    # Base: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    candidates = set()
    
    # Generate base patterns from tetracode with various tails
    base_patterns = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            base_patterns.append(h + list(tail))
    
    # Apply all 720 coordinate permutations to base patterns
    # But to keep it manageable, apply a subset of high-symmetry permutations
    # Actually, let's apply ALL permutations but be smart about it
    
    # First, collect unique base patterns
    base_set = set(tuple(p) for p in base_patterns)
    
    # Apply coordinate permutations
    perm_candidates = set()
    for perm in itertools.permutations(range(n)):
        for bp in base_set:
            permuted = [bp[perm[i]] for i in range(n)]
            perm_candidates.add(tuple(permuted))
    
    # Apply value shifts (add c mod 3 to all positions)
    shifted_candidates = set()
    for vc in range(3):
        for pc in perm_candidates:
            shifted = [(pc[i] + vc) % 3 for i in range(n)]
            shifted_candidates.add(tuple(shifted))
    
    # Apply value permutation (swap 1 and 2)
    valperm_candidates = set()
    for pc in shifted_candidates:
        valperm = [0 if pc[i] == 0 else (3 - pc[i]) for i in range(n)]
        valperm_candidates.add(tuple(valperm))
    
    # Combine all
    all_candidates = perm_candidates | shifted_candidates | valperm_candidates
    
    # Also add the original tetracode words
    for bp in base_set:
        all_candidates.add(bp)
    
    return list(all_candidates)


def _fast_greedy_set_cover(candidates, M, total, time_limit=30.0):
    """
    Fast greedy set cover using NumPy vectorized operations.
    candidates: list of word indices
    M: coverage matrix
    """
    start_time = time.time()
    candidate_indices = list(set(candidates))
    
    if not candidate_indices:
        return []
    
    # Precompute coverage vectors for candidates
    n_cands = len(candidate_indices)
    cov_matrix = M[candidate_indices, :]  # shape (n_cands, total)
    
    uncovered = np.ones(total, dtype=bool)
    selected = []
    uncovered_count = total
    
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
        
        # Count how many uncovered words each candidate covers
        # Vectorized: for each candidate, count True values in uncovered positions
        # cov_matrix is (n_cands, total), uncovered is (total,)
        # We want: for each row, sum of (cov_matrix[i,j] & uncovered[j])
        
        # Efficient approach: cov_matrix[:, uncovered] gives (n_cands, n_uncovered)
        # Then sum along axis 1
        if uncovered_count < total:
            unc_idx = np.where(uncovered)[0]
            counts = cov_matrix[:, unc_idx].sum(axis=1)
        else:
            counts = cov_matrix.sum(axis=1)
        
        best_local = np.argmax(counts)
        best_global = candidate_indices[best_local]
        
        if counts[best_local] == 0:
            # No candidate can cover more uncovered words; break
            break
        
        selected.append(best_global)
        
        # Update uncovered
        cov_vec = M[best_global]
        newly_covered = cov_vec & uncovered
        uncovered[~cov_vec] = False  # This is wrong, let me fix
        
        # Correct update:
        uncovered[cov_vec == 1] = False
        uncovered_count = uncovered.sum()
    
    return selected


def _verify_code(code_indices, M, total):
    """Verify that every word is covered."""
    if len(code_indices) == 0:
        return total == 0
    selected = np.zeros(total, dtype=bool)
    selected[code_indices] = True
    covered = (M @ selected) >= 1
    return covered.all()


def _local_search_polish(code_indices, M, all_words, total, time_limit=60.0):
    """
    Intensive local search: greedy deletion followed by random swaps.
    """
    code_indices = list(code_indices)
    start_time = time.time()
    
    # Maintain counts: how many codewords cover each word
    counts = np.zeros(total, dtype=np.int32)
    for idx in code_indices:
        counts[M[idx] == 1] += 1
        
    code_set = set(code_indices)
    rng = random.Random(42)
    
    def try_deletion():
        """Attempt to remove redundant codewords. Returns True if any removed."""
        nonlocal code_indices, counts, code_set
        changed = False
        # Iterate backwards to allow popping
        i = len(code_indices) - 1
        while i >= 0:
            idx = code_indices[i]
            cov = M[idx] == 1
            # If any word in this codeword's ball is covered ONLY by this codeword, we can't remove it
            if (counts[cov] == 1).any():
                i -= 1
                continue
            # Remove it
            counts[cov] -= 1
            code_indices.pop(i)
            code_set.remove(idx)
            changed = True
            # Don't decrement i; check same position again
        return changed

    # Initial greedy deletion
    while try_deletion():
        pass
        
    best_size = len(code_indices)
    
    while time.time() - start_time < time_limit:
        if len(code_indices) == 0:
            break
            
        # Try a random swap
        i = rng.randint(0, len(code_indices) - 1)
        idx_out = code_indices[i]
        cov_out = M[idx_out] == 1
        
        # Find a replacement
        idx_in = rng.randint(0, total - 1)
        if idx_in in code_set:
            continue
            
        cov_in = M[idx_in] == 1
        
        # Check if swap is valid
        risky = cov_out
        if not np.all(counts[risky] + cov_in[risky] >= 2):
            continue
        
        # Swap is valid. Perform it.
        counts[cov_out] -= 1
        counts[cov_in] += 1
        code_indices[i] = idx_in
        code_set.remove(idx_out)
        code_set.add(idx_in)
        
        # Try to delete redundant words after the swap
        deleted_any = try_deletion()
        
        if deleted_any:
            best_size = len(code_indices)
        else:
            # Revert
            counts[cov_in] -= 1
            counts[cov_out] += 1
            code_indices[i] = idx_out
            code_set.remove(idx_in)
            code_set.add(idx_out)
            
    return code_indices


def _greedy_construct(M, all_words, total):
    """
    Greedy set cover: at each step, pick the word that covers the most uncovered words.
    """
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        counts = M[:, uncovered].sum(axis=1)
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[M[best_idx] == 1] = False
    
    return selected


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    
    # Build coverage matrix
    M, all_words = _build_coverage_matrix(q, n, radius)
    
    # 1. Generate algebraic candidate pool
    candidates_words = _generate_algebraic_candidates(q, n, radius)
    
    # Convert to indices
    candidate_indices = []
    for w in candidates_words:
        idx = _word_to_idx(w, q, n)
        candidate_indices.append(idx)
    
    # 2. Fast greedy set cover on candidate pool
    elapsed = time.time() - start_time
    remaining_for_greedy = min(30.0, _DEADLINE - elapsed - 20)
    if remaining_for_greedy > 2:
        greedy_result = _fast_greedy_set_cover(candidate_indices, M, total, time_limit=remaining_for_greedy)
        if _verify_code(greedy_result, M, total):
            best_code = greedy_result
        else:
            # Fallback: use full greedy
            best_code = _greedy_construct(M, all_words, total)
    else:
        best_code = _greedy_construct(M, all_words, total)
    
    # 3. Local search polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10
    if remaining > 5:
        polished_code = _local_search_polish(best_code, M, all_words, total, time_limit=remaining)
        if len(polished_code) < len(best_code):
            best_code = polished_code
    
    # 4. Final verification and fallback
    if not _verify_code(best_code, M, total):
        greedy_code = _greedy_construct(M, all_words, total)
        return [all_words[i].tolist() for i in greedy_code]
    
    return [all_words[i].tolist() for i in best_code]
