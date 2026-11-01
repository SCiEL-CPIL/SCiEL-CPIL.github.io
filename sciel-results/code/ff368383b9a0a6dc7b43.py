"""Improved superpermutation constructor with better overlap handling and time management."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

def _max_overlap_fast(a, b):
    """Find maximum k such that suffix of a of length k == prefix of b of length k."""
    max_k = min(len(a), len(b))
    # Try from largest possible overlap down
    for k in range(max_k, -1, -1):
        if k == 0:
            return 0
        if a[-k:] == b[:k]:
            return k
    return 0

def _build_word_from_order(perms, m):
    """Build the word for step m given an ordering of (m-1)-permutations."""
    if not perms:
        return []
    
    # Precompute blocks
    blocks = [list(p) + [m] + list(p) for p in perms]
    
    # Build word greedily with maximum overlap
    word = list(blocks[0])
    for i in range(1, len(blocks)):
        prev = blocks[i-1]
        curr = blocks[i]
        ov = _max_overlap_fast(prev, curr)
        word.extend(curr[ov:])
    
    return word

def _try_orderings(perms, m, time_limit):
    """Try different orderings of perms to find one that gives shortest word."""
    if len(perms) <= 1:
        return _build_word_from_order(perms, m)
    
    start_time = time.time()
    
    # Start with natural ordering
    best_word = _build_word_from_order(perms, m)
    best_len = len(best_word)
    
    # Try some heuristic orderings
    # Ordering by last element
    perms_by_last = sorted(perms, key=lambda p: p[-1])
    if perms_by_last != perms:
        word = _build_word_from_order(perms_by_last, m)
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    # Ordering by first element
    perms_by_first = sorted(perms, key=lambda p: p[0])
    if perms_by_first != perms:
        word = _build_word_from_order(perms_by_first, m)
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    # Try reverse ordering
    perms_rev = list(reversed(perms))
    if perms_rev != perms:
        word = _build_word_from_order(perms_rev, m)
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    # For small cases, try more exhaustive search
    if len(perms) <= 8 and (time.time() - start_time) < time_limit * 0.5:
        # Try all permutations for very small cases
        best_perm_order = None
        for perm_order in itertools.permutations(range(len(perms))):
            if (time.time() - start_time) >= time_limit * 0.7:
                break
            ordered = [perms[i] for i in perm_order]
            word = _build_word_from_order(ordered, m)
            if len(word) < best_len:
                best_len = len(word)
                best_word = word
    
    return best_word

def construct(n):
    """Construct a valid superpermutation for n symbols using a greedy algorithm."""
    start_time = time.time()
    deadline = start_time + 140
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341214321341243124132431421"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # Greedy construction
    # Start with the first permutation
    current = list(perms[0])
    remaining = set(perms[1:])
    
    # Map suffixes to possible next permutations
    # We want to find the permutation that maximizes overlap with the current suffix
    
    while remaining:
        # Find the best next permutation
        best_perm = None
        best_overlap = 0
        
        # Check time
        if time.time() > deadline - 5:
            # Fallback: just append the rest (will be very long but valid if we ensure all are added)
            # This fallback is dangerous for validity if we don't add all.
            # Instead, we break and return what we have? No, must be valid.
            # We must finish.
            pass
        
        for p in remaining:
            # Calculate max overlap between current suffix and p prefix
            max_k = min(len(current), n)
            for k in range(max_k, 0, -1):
                if current[-k:] == list(p[:k]):
                    if k > best_overlap:
                        best_overlap = k
                        best_perm = p
                    break
        
        if best_perm is None:
            # No overlap, just append
            best_perm = remaining.pop()
            current.extend(best_perm)
        else:
            remaining.remove(best_perm)
            # Append the non-overlapping part
            current.extend(best_perm[best_overlap:])
    
    return "".join(map(str, current))
# EVOLVE-BLOCK-END
