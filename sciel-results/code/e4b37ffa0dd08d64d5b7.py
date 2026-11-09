"""Covering code search for K_q(n, radius) using Tetracode initialization + intensive local search."""
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


def _verify_code(code_indices, M, total):
    """Verify that every word is covered."""
    if len(code_indices) == 0:
        return total == 0
    selected = np.zeros(total, dtype=bool)
    selected[code_indices] = True
    # Check if M @ selected >= 1 for all rows
    covered = (M @ selected) >= 1
    return covered.all()


def _local_search_polish(code_indices, M, all_words, total, time_limit=60.0):
    """
    Intensive local search: greedy deletion followed by simulated annealing / random swaps.
    """
    code_indices = list(code_indices)
    start_time = time.time()
    
    # Precompute covers as boolean arrays for speed
    covers = [M[idx] == 1 for idx in code_indices]
    
    # Maintain counts: how many codewords cover each word
    counts = np.zeros(total, dtype=np.int32)
    for cov in covers:
        counts[cov] += 1
        
    code_set = set(code_indices)
    rng = random.Random(42)
    
    def try_deletion():
        """Attempt to remove redundant codewords. Returns True if any removed."""
        nonlocal code_indices, covers, counts, code_set
        changed = False
        # Iterate backwards to allow popping
        for i in range(len(code_indices) - 1, -1, -1):
            cov = covers[i]
            # If any word in this codeword's ball is covered ONLY by this codeword, we can't remove it
            if (counts[cov] == 1).any():
                continue
            # Remove it
            counts[cov] -= 1
            removed_idx = code_indices.pop(i)
            covers.pop(i)
            code_set.remove(removed_idx)
            changed = True
        return changed

    # Initial greedy deletion
    while try_deletion():
        pass
        
    best_size = len(code_indices)
    no_improvement_count = 0
    temperature = 10.0  # Initial "temperature" for accepting non-improving moves? 
                        # Actually, we only accept improving moves (size reduction) or equal size 
                        # moves that lead to future reductions? 
                        # Let's stick to strict improvement (size reduction) via swaps that enable deletions.
    
    while time.time() - start_time < time_limit:
        # 1. Try to find a swap that reduces the size
        # Strategy: Pick a random codeword to remove. If it's essential, find a replacement that covers its unique parts.
        # Then try to delete others.
        
        # Simple heuristic: Random swap
        if len(code_indices) > 0:
            i = rng.randint(0, len(code_indices) - 1)
            idx_out = code_indices[i]
            cov_out = covers[i]
            
            # Find a replacement idx_in
            # We want idx_in to cover the "unique" parts of idx_out, or just be a good candidate.
            # Random selection is slow. Let's pick a random word from the space.
            idx_in = rng.randint(0, total - 1)
            if idx_in in code_set:
                continue
                
            cov_in = M[idx_in] == 1
            
            # Check if swap is valid (all words still covered)
            # New count for w in cov_out: counts[w] - 1 + (1 if w in cov_in else 0)
            # We need new_count >= 1
            # i.e., counts[w] + cov_in[w] >= 2 for all w in cov_out
            
            # Efficient check
            risky = cov_out
            if not np.all(counts[risky] + cov_in[risky] >= 2):
                continue
            
            # Swap is valid. Perform it.
            counts[cov_out] -= 1
            counts[cov_in] += 1
            code_indices[i] = idx_in
            covers[i] = cov_in
            code_set.remove(idx_out)
            code_set.add(idx_in)
            
            # Try to delete redundant words after the swap
            deleted_any = try_deletion()
            
            if deleted_any:
                # We improved (size decreased)
                best_size = len(code_indices)
                no_improvement_count = 0
            else:
                # No immediate improvement. 
                # We can either revert or keep the swap to explore the neighborhood.
                # Keeping the swap allows us to escape local minima if we use a more complex search.
                # For now, let's revert if no improvement to stay close to good solutions?
                # Or keep it? If we keep it, we might wander.
                # Let's revert to maintain a "greedy" bias towards smaller sets.
                # Undo swap
                counts[cov_in] -= 1
                counts[cov_out] += 1
                code_indices[i] = idx_out
                covers[i] = cov_out
                code_set.remove(idx_in)
                code_set.add(idx_out)
                
                no_improvement_count += 1
                if no_improvement_count > 1000:
                    # Restart local search with a different seed or break?
                    # Let's just continue, maybe a random walk will find a path.
                    # But to be safe, let's try a "kick": randomly swap 2 words and see if it helps.
                    pass
        else:
            break
            
    return code_indices

def _greedy_construct(M, all_words, total):
    """
    Greedy set cover: at each step, pick the word that covers the most uncovered words.
    """
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        # Count how many uncovered words each candidate covers
        counts = M[:, uncovered].sum(axis=1)
        # Pick the one with max count
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
    
    # 1. Try Tetracode initialization if applicable
    tetracode_code = _direct_sum_tetracode(q, n, radius)
    if tetracode_code is not None:
        # Convert words to indices
        tetracode_indices = []
        for w in tetracode_code:
            idx = _word_to_idx(w, q, n)
            tetracode_indices.append(idx)
        best_code = tetracode_indices
        best_size = len(best_code)
    else:
        # Fallback to greedy if tetracode not applicable
        greedy_code = _greedy_construct(M, all_words, total)
        best_code = greedy_code
        best_size = len(best_code)
    
    # 2. Intensive local search polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10  # Leave 10s margin
    if remaining > 5:
        polished_code = _local_search_polish(best_code, M, all_words, total, time_limit=remaining)
        if len(polished_code) < best_size:
            best_code = polished_code
            best_size = len(best_code)
    
    # 3. If we still have time and haven't reached the theoretical lower bound (71) or known best (73), 
    #    try a second round with a different random seed or slightly different strategy?
    #    For now, the single intensive search is the main driver.
    
    # Convert indices back to words
    best_code_words = [all_words[i].tolist() for i in best_code]
    
    # Final verification
    if not _verify_code(best_code, M, total):
        # Fallback: use a valid greedy solution
        greedy_code = _greedy_construct(M, all_words, total)
        return [all_words[i].tolist() for i in greedy_code]
    
    return best_code_words
