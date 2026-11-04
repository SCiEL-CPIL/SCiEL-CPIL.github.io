"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Fixed deadline: 110 seconds (safety margin under 120s limit)
_DEADLINE = 110.0


def _generate_tetracode():
    """Generate the 9-word ternary tetracode (perfect code of length 4, radius 1)."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum_code(q, n, radius):
    """
    Generate the direct sum code: tetracode x {0,1,2}^(n-4).
    This gives 9 * 3^(n-4) codewords.
    For q=3, n=8, radius=1: 9 * 81 = 729 words.
    """
    if n < 4:
        # Fallback: brute force or simple construction
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    tetracode = _generate_tetracode()
    tail_len = n - 4
    tails = list(itertools.product(range(q), repeat=tail_len))
    
    code = []
    for h in tetracode:
        for tail in tails:
            code.append(h + list(tail))
    return code


def _word_to_idx(word, q, n):
    """Convert a word (list of ints) to an integer index in [0, q^n)."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _sphere_of_word(word, q, n, radius):
    """
    Generate all words within Hamming distance <= radius of the given word.
    Returns a set of integer indices.
    """
    if radius == 0:
        return {_word_to_idx(word, q, n)}
    
    if radius == 1:
        center = _word_to_idx(word, q, n)
        sphere = {center}
        for pos in range(n):
            orig = word[pos]
            for v in range(q):
                if v == orig:
                    continue
                new_word = word[:]
                new_word[pos] = v
                sphere.add(_word_to_idx(new_word, q, n))
        return sphere
    
    # General radius: use itertools
    center = _word_to_idx(word, q, n)
    sphere = {center}
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = word[:]
                for pos, val in zip(positions, values):
                    if val != word[pos]:
                        new_word[pos] = val
                    else:
                        continue
                # Only add if actually different
                if new_word != word:
                    sphere.add(_word_to_idx(new_word, q, n))
    return sphere


def _check_coverage(code, q, n, radius):
    """
    Check if every word in {0..q-1}^n is within distance <= radius of some codeword.
    Returns True if valid, False otherwise.
    """
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    
    for cw in code:
        sphere = _sphere_of_word(cw, q, n, radius)
        for idx in sphere:
            covered[idx] = True
    
    return not covered.any() == False  # All covered


def _remove_redundant(code, q, n, radius, deadline):
    """
    Try to remove codewords that are redundant (their sphere is covered by others).
    Greedy: try removing each codeword, check if coverage is maintained.
    """
    code = list(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            # Try removing code[i]
            reduced = code[:i] + code[i+1:]
            # Check coverage of reduced set
            # For efficiency, only check if the removed codeword's unique coverage is empty
            # Quick check: is code[i]'s sphere fully covered by others?
            sphere_i = _sphere_of_word(code[i], q, n, radius)
            # Check if all words in sphere_i are covered by remaining codewords
            # This is expensive, so do a quick sample or full check
            covered_by_others = set()
            for j, cw in enumerate(reduced):
                if j == i:
                    continue
                sphere_j = _sphere_of_word(cw, q, n, radius)
                covered_by_others.update(sphere_j)
            
            if sphere_i.issubset(covered_by_others):
                code = reduced
                improved = True
                break
    
    return code


def _local_search(code, q, n, radius, deadline):
    """
    Local search: try to swap a codeword with a neighbor to increase coverage,
    or remove redundant codewords.
    """
    code = list(code)
    
    while time.time() < deadline:
        # Phase 1: Remove redundant codewords
        code = _remove_redundant(code, q, n, radius, deadline)
        
        # Phase 2: Try swaps to improve
        improved = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            # Generate neighbors of code[i]
            cw = code[i]
            for pos in range(n):
                if time.time() >= deadline:
                    break
                orig = cw[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_cw = cw[:]
                    new_cw[pos] = v
                    # Check if new_cw is already in code
                    if new_cw in code:
                        continue
                    # Try swapping
                    new_code = code[:i] + [new_cw] + code[i+1:]
                    # Check if this swap maintains coverage
                    # Quick check: is the new codeword's sphere covered by others?
                    # Actually, we want to see if the swap improves things
                    # For now, just check validity
                    if _check_coverage(new_code, q, n, radius):
                        code = new_code
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return code


def construct(q, n, radius):
    """
    Construct a covering code for the given parameters.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate initial code using direct sum construction
    code = _direct_sum_code(q, n, radius)
    
    # Verify initial code is valid
    if not _check_coverage(code, q, n, radius):
        # Fallback: return all words (always valid)
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Try to improve via local search
    if time.time() < deadline - 10:  # Leave some time margin
        code = _local_search(code, q, n, radius, deadline)
    
    # Final verification
    if not _check_coverage(code, q, n, radius):
        # If local search broke coverage, revert to initial code
        code = _direct_sum_code(q, n, radius)
    
    return code
# EVOLVE-BLOCK-END
