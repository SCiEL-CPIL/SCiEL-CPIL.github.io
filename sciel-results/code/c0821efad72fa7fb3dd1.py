"""Superpermutation search: classic construction + optimized local search with count tracking."""
# EVOLVE-BLOCK-START
import time
import sys
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


def _init_counts(word, n, perm_to_id):
    """Initialize count array: for each permutation, how many windows cover it."""
    num_perms = _factorial(n)
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    return counts


def _count_zero(counts):
    """Count how many permutations have count == 0."""
    z = 0
    for c in counts:
        if c == 0:
            z += 1
    return z


def _try_remove_chars_fast(word, n, time_limit):
    """
    Shorten superpermutation using fast count-tracking deletion.
    Each deletion attempt is O(n) instead of O(n * L).
    Uses a zero-count tracker for O(1) validity check.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    zero_count = _count_zero(counts)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    
    # Fast deletion: try removing each position, use incremental count updates
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            if len(word) - 1 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include position i
            # Old windows containing position i: start k where k <= i < k+n
            # i.e., k in [max(0, i-n+1), min(i, L-n)]
            k_start = max(0, i - n + 1)
            k_end = min(i, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
            
            # Step 2: Compute new windows formed after deletion
            # After removing position i, new word has length L-1
            # New windows affected: start k in [max(0, i-n+1), min(i-1, L-1-n)] in new word
            # Map to old word indices
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 1 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                # In new word, window starting at k
                # If k < i: new_word[k:k+n] = word[k:i] + word[i+1:k+n+1]
                # If k >= i: new_word[k:k+n] = word[k+1:k+1+n]
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 1:k + n + 1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 1:k + 1 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            if zero_count == 0:
                # Accept deletion
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                # Restore old windows
                for k, pid in old_counts:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        zero_count -= 1
                # Remove new windows
                for pid in new_pids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
    
    return word


def _try_remove_chars_forward(word, n, time_limit):
    """
    Try deletion in forward order for additional passes.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    zero_count = _count_zero(counts)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    
    # Fast deletion: try removing each position in forward order
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word)):
            if time.time() > deadline - 2:
                break
            if len(word) - 1 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include position i
            k_start = max(0, i - n + 1)
            k_end = min(i, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
            
            # Step 2: Compute new windows formed after deletion
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 1 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 1:k + n + 1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 1:k + 1 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            if zero_count == 0:
                # Accept deletion
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                for k, pid in old_counts:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        zero_count -= 1
                for pid in new_pids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
    
    return word


def _try_double_delete(word, n, time_limit):
    """
    Try deleting two adjacent characters at once.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    zero_count = _count_zero(counts)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word) - 2, -1, -1):
            if time.time() > deadline - 2:
                break
            if len(word) - 2 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include position i or i+1
            # Windows containing i or i+1: start k where k <= i < k+n OR k <= i+1 < k+n
            # i.e., k in [max(0, i-n+1), min(i+1, L-n)]
            k_start = max(0, i - n + 1)
            k_end = min(i + 1, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
            
            # Step 2: Compute new windows formed after deleting positions i and i+1
            # New word has length L-2
            # New windows affected: start k in [max(0, i-n+1), min(i-1, L-2-n)] in new word
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 2 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                # In new word, window starting at k
                # If k < i: new_word[k:k+n] = word[k:i] + word[i+2:k+n+2]
                # If k >= i: new_word[k:k+n] = word[k+2:k+2+n]
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 2:k + n + 2]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 2:k + 2 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            if zero_count == 0:
                # Accept deletion
                del word[i+1]
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                for k, pid in old_counts:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        zero_count -= 1
                for pid in new_pids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
    
    return word


def _try_swap_then_delete(word, n, time_limit):
    """
    Try swapping two characters to enable subsequent deletions.
    This is a more aggressive local search move.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    zero_count = _count_zero(counts)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    best_word = list(word)
    
    # Try swapping pairs that are close together
    L = len(word)
    # Limit the range of swaps to keep it tractable
    max_swap_dist = min(50, L // 10)
    
    for i in range(L):
        if time.time() > deadline - 5:
            break
        for j in range(i + 1, min(i + max_swap_dist, L)):
            if time.time() > deadline - 5:
                break
            
            # Save state
            old_a = word[i]
            old_b = word[j]
            if old_a == old_b:
                continue
            
            # Perform swap
            word[i] = old_b
            word[j] = old_a
            
            # Recompute counts (expensive but necessary for correctness after swap)
            new_counts = _init_counts(word, n, perm_to_id)
            new_zero = _count_zero(new_counts)
            
            if new_zero == 0:
                # Valid after swap, try to delete
                # Try deleting each position
                for k in range(len(word) - 1, -1, -1):
                    if len(word) - 1 < n:
                        break
                    
                    # Check if we can delete position k
                    L2 = len(word)
                    k_start = max(0, k - n + 1)
                    k_end = min(k, L2 - n)
                    
                    temp_counts = list(new_counts)
                    valid = True
                    
                    for kk in range(k_start, k_end + 1):
                        p = tuple(word[kk:kk + n])
                        pid = perm_to_id.get(p)
                        if pid is not None:
                            temp_counts[pid] -= 1
                            if temp_counts[pid] == 0:
                                # Check if this permutation is still covered by other windows
                                # Simplified: just check if it becomes 0
                                pass
                    
                    # This approach is getting complex. Let's use a simpler heuristic.
                    # Just try the deletion and check validity
                    old_word = list(word)
                    del word[k]
                    
                    check_counts = _init_counts(word, n, perm_to_id)
                    check_zero = _count_zero(check_counts)
                    
                    if check_zero == 0:
                        if len(word) < best_len:
                            best_len = len(word)
                            best_word = list(word)
                    else:
                        word = old_word
                    
                    if time.time() > deadline - 5:
                        break
            else:
                # Invalid after swap, revert
                word[i] = old_a
                word[j] = old_b
    
    if best_len < len(best_word):
        return best_word
    return best_word


def _try_triple_delete(word, n, time_limit):
    """
    Try deleting three adjacent characters at once.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    zero_count = _count_zero(counts)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word) - 3, -1, -1):
            if time.time() > deadline - 2:
                break
            if len(word) - 3 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include positions i, i+1, or i+2
            # Windows containing any of i, i+1, i+2: start k where k <= i < k+n OR k <= i+1 < k+n OR k <= i+2 < k+n
            # i.e., k in [max(0, i-n+1), min(i+2, L-n)]
            k_start = max(0, i - n + 1)
            k_end = min(i + 2, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
            
            # Step 2: Compute new windows formed after deleting positions i, i+1, and i+2
            # New word has length L-3
            # New windows affected: start k in [max(0, i-n+1), min(i-1, L-3-n)] in new word
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 3 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                # In new word, window starting at k
                # If k < i: new_word[k:k+n] = word[k:i] + word[i+3:k+n+3]
                # If k >= i: new_word[k:k+n] = word[k+3:k+3+n]
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 3:k + n + 3]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 3:k + 3 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            if zero_count == 0:
                # Accept deletion
                del word[i+2]
                del word[i+1]
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                for k, pid in old_counts:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        zero_count -= 1
                for pid in new_pids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
    
    return word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 120  # evaluator limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Step 1: Build classic superpermutation
    word = _classic_superperm(n)
    initial_len = len(word)
    
    # Step 2: Try to shorten it with fast count-tracking (backward pass)
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 20  # safety margin
    
    if remaining > 10:
        word = _try_remove_chars_fast(word, n, remaining // 3)
    
    # Step 3: Try forward pass
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 15
    
    if remaining > 10:
        word = _try_remove_chars_forward(word, n, remaining // 3)
    
    # Step 4: Try double deletion
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 10
    
    if remaining > 10:
        word = _try_double_delete(word, n, remaining // 3)
    
    # Step 5: Try triple deletion
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 8
    
    if remaining > 8:
        word = _try_triple_delete(word, n, remaining // 2)
    
    # Step 6: Another backward pass to catch any new opportunities
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 5
    
    if remaining > 5:
        word = _try_remove_chars_fast(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
