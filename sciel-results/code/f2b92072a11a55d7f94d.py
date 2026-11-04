"""Structural change: greedy set-cover with bitset acceleration + local improvement,
plus a deadline guard. Replaces the fixed direct-sum construction with an adaptive
search that can beat 486 for (q=3, n=8, radius=1) if time permits."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# ---- Bitset helpers for fast Hamming-distance-1 coverage checks ----

def _build_distance1_bitsets(q, n, radius):
    """
    For each word index w in [0, q**n), build a bitset (numpy array of uint64)
    marking which codeword positions are within Hamming distance <= radius of w.
    But we don't know codewords yet. Instead, we precompute for each coordinate i
    and value v, the set of word indices that have value v at position i.
    
    Actually, for greedy: given a candidate codeword c, the words it covers are
    all words within distance <= radius. For radius=1, that's c itself plus all
    words differing in exactly one position.
    
    We precompute: for each word w (as integer), a list of the words within
    distance 1. But storing all 6561 * 25 neighbors is fine.
    
    Better approach: represent the universe of 6561 words as a bitset.
    For each possible codeword c (there are 6561), precompute the bitset of
    words it covers.
    """
    total = q ** n
    # neighbors[w] = list of words within distance <= radius from w
    # For radius=1: w itself + all single-position changes
    neighbors = []
    for w in range(total):
        # Convert w to base-q digits
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits = digits[::-1]  # most significant first
        
        covered = {w}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == digits[pos]:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    # Convert back to integer
                    new_w = 0
                    for d in new_digits:
                        new_w = new_w * q + d
                    covered.add(new_w)
        neighbors.append(covered)
    
    # Build bitsets: coverage[c] = numpy uint64 array representing set of words covered by codeword c
    # total words = q**n. For q=3, n=8: 6561 words. 6561 bits = 103 uint64s.
    num_blocks = (total + 63) // 64
    coverage = np.zeros((total, num_blocks), dtype=np.uint64)
    for c in range(total):
        for w in neighbors[c]:
            coverage[c, w // 64] |= (1 << (w % 64))
    
    return coverage, neighbors


def _bitset_union(a, b):
    """Return a | b for two bitsets (numpy uint64 arrays)."""
    return a | b


def _bitset_count(a):
    """Count bits set in bitset a."""
    return np.unpackbits(a.view(np.uint8)).sum()


def _greedy_cover(q, n, radius, deadline):
    """
    Greedy set cover: repeatedly pick the codeword that covers the most uncovered words.
    """
    total = q ** n
    coverage, neighbors = _build_distance1_bitsets(q, n, radius)
    
    # uncovered: bitset of all words
    num_blocks = (total + 63) // 64
    uncovered = np.ones(num_blocks, dtype=np.uint64)
    # Mask off extra bits in last block
    extra = num_blocks * 64 - total
    if extra > 0:
        uncovered[-1] &= (1 << (64 - extra)) - 1
    
    codewords = []
    start_time = time.time()
    
    while _bitset_count(uncovered) > 0:
        if time.time() > deadline:
            break
        
        # Find codeword with max coverage of uncovered
        best_c = -1
        best_count = -1
        for c in range(total):
            covered_mask = coverage[c] & uncovered
            cnt = _bitset_count(covered_mask)
            if cnt > best_count:
                best_count = cnt
                best_c = c
                if best_count == _bitset_count(uncovered):
                    break  # Can't do better
        
        if best_c < 0:
            break
        
        # Add codeword
        digits = []
        temp = best_c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits = digits[::-1]
        codewords.append(digits)
        
        # Update uncovered
        uncovered &= ~coverage[best_c]
    
    return codewords


def _local_improve(codewords, q, n, radius, deadline):
    """
    Try to remove redundant codewords.
    A codeword is redundant if removing it still leaves all words covered.
    """
    total = q ** n
    coverage, neighbors = _build_distance1_bitsets(q, n, radius)
    num_blocks = (total + 63) // 64
    
    # Build coverage bitset for current codewords
    covered = np.zeros(num_blocks, dtype=np.uint64)
    for c in codewords:
        # Convert c to index
        idx = 0
        for d in c:
            idx = idx * q + d
        covered |= coverage[idx]
    
    # Check if fully covered
    all_words = np.ones(num_blocks, dtype=np.uint64)
    extra = num_blocks * 64 - total
    if extra > 0:
        all_words[-1] &= (1 << (64 - extra)) - 1
    
    if (covered & all_words).sum() != all_words.sum() or not np.array_equal(covered & all_words, all_words):
        # Not fully covered, can't remove anything
        return codewords
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = codewords[i]
            idx = 0
            for d in c:
                idx = idx * q + d
            # Check if removing this codeword still covers all
            new_covered = covered & ~coverage[idx]
            if np.array_equal(new_covered & all_words, all_words):
                # Redundant, remove
                covered = new_covered
                codewords.pop(i)
                improved = True
                break  # Restart scan
    
    return codewords


def _swap_improve(codewords, q, n, radius, deadline):
    """
    Try to replace one codeword with another to enable removal of a second.
    Simplified: try adding a new codeword, then see if any existing becomes redundant.
    """
    total = q ** n
    coverage, neighbors = _build_distance1_bitsets(q, n, radius)
    num_blocks = (total + 63) // 64
    
    all_words = np.ones(num_blocks, dtype=np.uint64)
    extra = num_blocks * 64 - total
    if extra > 0:
        all_words[-1] &= (1 << (64 - extra)) - 1
    
    # Current coverage
    covered = np.zeros(num_blocks, dtype=np.uint64)
    for c in codewords:
        idx = 0
        for d in c:
            idx = idx * q + d
        covered |= coverage[idx]
    
    if not np.array_equal(covered & all_words, all_words):
        return codewords
    
    # Try adding each candidate and see if we can then remove two
    current_set = set()
    for c in codewords:
        current_set.add(tuple(c))
    
    improved = False
    for cand_idx in range(total):
        if time.time() > deadline:
            break
        cand_digits = []
        temp = cand_idx
        for _ in range(n):
            cand_digits.append(temp % q)
            temp //= q
        cand_digits = cand_digits[::-1]
        if tuple(cand_digits) in current_set:
            continue
        
        # Try adding cand
        new_covered = covered | coverage[cand_idx]
        if not np.array_equal(new_covered & all_words, all_words):
            continue  # Doesn't help cover
        
        # Now try to remove two codewords
        new_list = codewords + [cand_digits]
        found_removal = False
        for i in range(len(new_list) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = new_list[i]
            idx = 0
            for d in c:
                idx = idx * q + d
            test_covered = new_covered & ~coverage[idx]
            if np.array_equal(test_covered & all_words, all_words):
                # Remove this one, then try to remove another
                new_list.pop(i)
                test_covered2 = test_covered
                for j in range(len(new_list) - 1, -1, -1):
                    c2 = new_list[j]
                    idx2 = 0
                    for d in c2:
                        idx2 = idx2 * q + d
                    test_covered3 = test_covered2 & ~coverage[idx2]
                    if np.array_equal(test_covered3 & all_words, all_words):
                        new_list.pop(j)
                        found_removal = True
                        break
                if found_removal:
                    codewords = new_list
                    covered = test_covered3
                    current_set = set(tuple(c) for c in codewords)
                    improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Handle edge cases
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    
    # For small cases, use greedy
    total = q ** n
    if total > 100000:
        # Too large for full bitset approach; fall back to tetracode direct sum
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Greedy set cover
    codewords = _greedy_cover(q, n, radius, deadline)
    
    # Local improvement: remove redundant
    if time.time() < deadline - 5:
        codewords = _local_improve(codewords, q, n, radius, deadline)
    
    # Swap improvement
    if time.time() < deadline - 5:
        codewords = _swap_improve(codewords, q, n, radius, deadline)
    
    # Final check: ensure validity
    # (The evaluator will check, but let's be safe)
    if time.time() < deadline - 2:
        coverage, _ = _build_distance1_bitsets(q, n, radius)
        num_blocks = (total + 63) // 64
        all_words = np.ones(num_blocks, dtype=np.uint64)
        extra = num_blocks * 64 - total
        if extra > 0:
            all_words[-1] &= (1 << (64 - extra)) - 1
        covered = np.zeros(num_blocks, dtype=np.uint64)
        for c in codewords:
            idx = 0
            for d in c:
                idx = idx * q + d
            covered |= coverage[idx]
        if not np.array_equal(covered & all_words, all_words):
            # Fallback to tetracode
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return codewords
# EVOLVE-BLOCK-END
