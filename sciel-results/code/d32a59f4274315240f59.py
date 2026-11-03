"""Superpermutation search: classic construction + fast local search with count tracking."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations


def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    return word


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _build_perm_index(n):
    """Build a dict mapping permutation tuples to unique IDs, and return list of all perms."""
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    return all_perms, perm_to_id


def _init_counts_and_missing(word, n, perm_to_id):
    """Initialize count array and missing counter."""
    num_perms = _factorial(n)
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    missing = sum(1 for c in counts if c == 0)
    return counts, missing


def _try_remove_chars_optimized(word, n, time_limit):
    """
    Shorten superpermutation using fast count-tracking deletion with O(1) validity checks.
    Uses missing counter to avoid scanning all 5040 permutations on each check.
    Multiple passes with different orderings to escape local optima.
    """
    word = list(word)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = _factorial(n)
    
    # Initialize counts and missing counter
    counts, missing = _init_counts_and_missing(word, n, perm_to_id)
    
    if missing > 0:
        return word
    
    best_len = len(word)
    best_word = list(word)
    
    def _try_deletion_at(word, i, n, counts, missing, perm_to_id):
        """
        Try deleting position i. Returns (success, new_missing, rollback_data).
        rollback_data is needed to undo the change.
        """
        L = len(word)
        
        # Old windows containing position i: k in [max(0, i-n+1), min(i, L-n)]
        k_start = max(0, i - n + 1)
        k_end = min(i, L - n)
        
        # Track changes for rollback
        changes = []  # list of (pid, delta)
        new_missing = missing
        
        # Step 1: Decrement counts for old windows
        for k in range(k_start, k_end + 1):
            p = tuple(word[k:k + n])
            pid = perm_to_id.get(p)
            if pid is not None:
                old_count = counts[pid]
                counts[pid] = old_count - 1
                changes.append((pid, -1))
                if old_count == 1 and counts[pid] == 0:
                    new_missing += 1
        
        # Step 2: Increment counts for new windows
        new_k_start = max(0, i - n + 1)
        new_k_end = min(i - 1, L - 1 - n)
        
        for k in range(new_k_start, new_k_end + 1):
            if k < i:
                w1 = word[k:i]
                w2 = word[i + 1:k + n + 1]
                p = tuple(w1 + w2)
            else:
                p = tuple(word[k + 1:k + 1 + n])
            
            pid = perm_to_id.get(p)
            if pid is not None:
                old_count = counts[pid]
                counts[pid] = old_count + 1
                changes.append((pid, 1))
                if old_count == 0 and counts[pid] == 1:
                    new_missing -= 1
        
        return new_missing, changes
    
    def _rollback(word, i, changes):
        """Rollback deletion at position i."""
        for pid, delta in changes:
            counts[pid] -= delta
    
    # Multiple passes with different orderings
    orderings = [
        ("backward", list(range(len(word) - 1, -1, -1))),
        ("forward", list(range(len(word)))),
    ]
    
    # Add some random orderings for diversity
    for _ in range(3):
        idx = list(range(len(word)))
        random.shuffle(idx)
        orderings.append(("random", idx))
    
    for pass_num, (name, order) in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            for i in order:
                if time.time() > deadline - 5:
                    break
                
                if i >= len(word) or len(word) - 1 < n:
                    continue
                
                # Try deletion
                new_missing, changes = _try_deletion_at(word, i, n, counts, missing, perm_to_id)
                
                if new_missing == 0:
                    # Accept deletion
                    del word[i]
                    missing = new_missing
                    improved = True
                    
                    # Update order list to remove deleted index and adjust later indices
                    if i < len(order):
                        order.pop(i)
                    # Adjust indices greater than i
                    for j in range(len(order)):
                        if order[j] > i:
                            order[j] -= 1
                else:
                    # Rollback
                    _rollback(word, i, changes)
        
        # Update best
        if len(word) < best_len:
            best_len = len(word)
            best_word = list(word)
    
    return best_word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 150  # evaluator limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Step 1: Build classic superpermutation
    word = _classic_superperm(n)
    
    # Step 2: Try to shorten it with optimized count-tracking
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 10  # safety margin
    
    if remaining > 5:
        word = _try_remove_chars_optimized(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
