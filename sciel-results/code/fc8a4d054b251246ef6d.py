"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_start_time = time.time()
_DEADLINE = 115.0  # Safety margin before 120s kill (evaluator limit)


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _precompute_neighbor_indices(q, n, radius):
    """
    For each word index 0..q^n-1, compute the list of indices of words
    within Hamming distance <= radius.
    Returns a list of arrays, each of length 1 + n*(q-1) for radius=1.
    """
    space_size = q ** n
    # Precompute powers of q
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i - 1] * q

    neighbor_lists = [None] * space_size

    for idx in range(space_size):
        # Get the word digits
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q

        nbrs = [idx]
        for pos in range(n):
            orig = digits[pos]
            p = powers[n - 1 - pos]
            for v in range(q):
                if v == orig:
                    continue
                nbrs.append(idx + (v - orig) * p)

        neighbor_lists[idx] = np.array(nbrs, dtype=np.int32)

    return neighbor_lists


def _greedy_cover_vectorized(q, n, radius, deadline):
    """
    Vectorized greedy set cover using numpy.
    Returns list of selected indices.
    """
    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)

    # Precompute a flat array of all neighbor indices for fast indexing
    nbr_len = 1 + n * (q - 1)
    nbr_flat = np.empty((space_size, nbr_len), dtype=np.int32)
    for i in range(space_size):
        nbr_flat[i] = nbrs[i]

    covered = np.zeros(space_size, dtype=np.bool_)
    selected = []
    uncovered_count = np.full(space_size, nbr_len, dtype=np.int32)

    while time.time() < deadline - 5:
        best_idx = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_idx]

        if best_count == 0:
            break

        selected.append(best_idx)
        covered[nbr_flat[best_idx]] = True

        uncovered_count = (~covered[nbr_flat]).sum(axis=1).astype(np.int32)

    return selected


def _multi_pass_prune(q, n, radius, selected, deadline):
    """
    Multi-pass reference-count pruning.
    Repeatedly removes any codeword whose entire coverage is redundant.
    """
    if not selected:
        return selected

    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)
    
    # Build cover sets for selected codewords
    cover_sets = [nbrs[i] for i in selected]
    
    # Reference count: how many codewords cover each word
    ref_count = np.zeros(space_size, dtype=np.int32)
    for cs in cover_sets:
        ref_count[cs] += 1
    
    active = [True] * len(selected)
    
    # Multiple passes to remove all redundant codewords
    for pass_num in range(10):
        if time.time() > deadline - 3:
            break
        any_removed = False
        for i in range(len(selected)):
            if not active[i]:
                continue
            cs = cover_sets[i]
            if np.any(ref_count[cs] == 1):
                continue  # Not removable
            # Removable
            active[i] = False
            ref_count[cs] -= 1
            any_removed = True
        if not any_removed:
            break
    
    return [selected[i] for i in range(len(selected)) if active[i]]


def _local_search_swap(q, n, radius, selected, deadline):
    """
    Local search: try to remove codewords and replace pairs with single better codewords.
    Returns improved list.
    """
    if not selected:
        return selected

    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)
    nbr_len = 1 + n * (q - 1)
    nbr_flat = np.empty((space_size, nbr_len), dtype=np.int32)
    for i in range(space_size):
        nbr_flat[i] = nbrs[i]

    # Build coverage bitmap
    covered = np.zeros(space_size, dtype=np.bool_)
    for idx in selected:
        covered[nbr_flat[idx]] = True

    # Precompute for each word, which selected codewords cover it (for fast removal checks)
    # Use a dictionary: word_idx -> list of selected indices that cover it
    word_to_coders = {}
    for si, idx in enumerate(selected):
        for w in nbr_flat[idx]:
            if w not in word_to_coders:
                word_to_coders[w] = []
            word_to_coders[w].append(si)

    best_selected = list(selected)
    best_covered = covered.copy()
    best_count = len(selected)

    # Pass 1: Try removing each codeword
    for si in range(len(best_selected)):
        if time.time() > deadline - 5:
            break
        idx = best_selected[si]
        # Check if removing idx leaves all its covered words still covered by others
        can_remove = True
        for w in nbr_flat[idx]:
            coders = word_to_coders.get(w, [])
            if len(coders) <= 1:
                can_remove = False
                break
        if can_remove:
            # Remove it
            best_selected.pop(si)
            # Update word_to_coders
            for w in nbr_flat[idx]:
                if si in word_to_coders[w]:
                    word_to_coders[w].remove(si)
            best_count -= 1
            si -= 1  # Re-check same index

    # Pass 2: Try swapping pairs (i,j) with a single codeword k that covers all words 
    # that only i and j cover
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        found_improvement = False
        for i in range(len(best_selected)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(best_selected)):
                if time.time() > deadline - 8:
                    break
                idx_i = best_selected[i]
                idx_j = best_selected[j]
                
                # Find words that are ONLY covered by i and/or j
                # A word w is "critical" if the only coders covering w are among {i, j}
                # We need to find a codeword k that covers all critical words
                # Critical words = words w where set(coders(w)) is subset of {i, j} and non-empty
                
                # This is expensive, so let's approximate: check if there's a k that covers
                # the union of words uniquely covered by i or j
                # Actually, let's just try: for each candidate k not in selected, 
                # check if replacing {i,j} with k still covers everything
                
                # Simpler heuristic: try to find if any single codeword covers all words
                # that would become uncovered if we remove both i and j
                words_only_i_or_j = []
                for w in nbr_flat[idx_i]:
                    coders = word_to_coders.get(w, [])
                    if all(c in (i, j) for c in coders) and len(coders) > 0:
                        words_only_i_or_j.append(w)
                for w in nbr_flat[idx_j]:
                    if w not in words_only_i_or_j:
                        coders = word_to_coders.get(w, [])
                        if all(c in (i, j) for c in coders) and len(coders) > 0:
                            words_only_i_or_j.append(w)
                
                if not words_only_i_or_j:
                    # Removing both is fine, no critical words
                    # Just remove both
                    best_selected.pop(j)
                    best_selected.pop(i)
                    # Update word_to_coders
                    for w in nbr_flat[idx_i]:
                        if i in word_to_coders[w]:
                            word_to_coders[w].remove(i)
                        if j in word_to_coders[w]:
                            word_to_coders[w].remove(j)
                    for w in nbr_flat[idx_j]:
                        if i in word_to_coders[w]:
                            word_to_coders[w].remove(i)
                        if j in word_to_coders[w]:
                            word_to_coders[w].remove(j)
                    best_count -= 2
                    found_improvement = True
                    break
                else:
                    # Need to find a k that covers all words_only_i_or_j
                    # Try a few candidates
                    for k_idx in range(space_size):
                        if k_idx in best_selected:
                            continue
                        # Check if k covers all critical words
                        k_covers = set(nbr_flat[k_idx])
                        if all(w in k_covers for w in words_only_i_or_j):
                            # Replace i and j with k
                            best_selected.pop(j)
                            best_selected.pop(i)
                            best_selected.append(k_idx)
                            # Update word_to_coders
                            for w in nbr_flat[idx_i]:
                                if i in word_to_coders[w]:
                                    word_to_coders[w].remove(i)
                                if j in word_to_coders[w]:
                                    word_to_coders[w].remove(j)
                            for w in nbr_flat[idx_j]:
                                if i in word_to_coders[w]:
                                    word_to_coders[w].remove(i)
                                if j in word_to_coders[w]:
                                    word_to_coders[w].remove(j)
                            for w in nbr_flat[k_idx]:
                                if k_idx not in word_to_coders:
                                    word_to_coders[w] = []
                                word_to_coders[w].append(len(best_selected) - 1)
                            best_count -= 1
                            found_improvement = True
                            break
                        if time.time() > deadline - 8:
                            break
                    if found_improvement:
                        break
            if found_improvement:
                break
        if not found_improvement:
            break

    return best_selected


def _verify_coverage(q, n, radius, selected):
    """Verify that all words are covered by the selected codewords."""
    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)
    covered = np.zeros(space_size, dtype=np.bool_)
    for idx in selected:
        covered[nbrs[idx]] = True
    return np.all(covered)


def _idx_to_word(idx, q, n):
    """Convert integer index to word list."""
    word = [0] * n
    temp = idx
    for i in range(n - 1, -1, -1):
        word[i] = temp % q
        temp //= q
    return word


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110.0

    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        return []

    # 1. Greedy cover (vectorized)
    selected = _greedy_cover_vectorized(q, n, radius, deadline)

    # 2. Multi-pass prune
    selected = _multi_pass_prune(q, n, radius, selected, deadline)

    # 3. Local search with swaps
    selected = _local_search_swap(q, n, radius, selected, deadline)

    # 4. Final prune
    selected = _multi_pass_prune(q, n, radius, selected, deadline)

    # 5. Verify coverage
    if not _verify_coverage(q, n, radius, selected):
        # Fallback to tetracode
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        return []

    # Convert indices to words
    result = [_idx_to_word(idx, q, n) for idx in selected]
    return result

# EVOLVE-BLOCK-END
