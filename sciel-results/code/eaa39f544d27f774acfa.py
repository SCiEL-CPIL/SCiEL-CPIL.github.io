# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

# Precompute permutation index for n=7 (and smaller)
_PERM_INDEX = {}
_PERM_LIST = []
_PERM_INV = []

def _build_perm_tables(n):
    global _PERM_INDEX, _PERM_LIST, _PERM_INV
    _PERM_INDEX = {}
    _PERM_LIST = []
    _PERM_INV = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        _PERM_INV.append(p)
        idx += 1

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_valid_superperm(word, n):
    if len(word) < n:
        return False
    total_perms = n * _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == total_perms

def _compute_window_perms(word, n):
    """Compute window_perm array: for each starting position, the perm index or -1."""
    total_perms = n * _factorial(n)
    L = len(word)
    num_windows = max(0, L - n + 1)
    window_perm = [-1] * num_windows
    for i in range(num_windows):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            idx = _PERM_INDEX.get(w, -1)
            if idx >= 0:
                window_perm[i] = idx
    return window_perm

def _compute_perm_counts(window_perm):
    total_perms = len(_PERM_INDEX)
    perm_count = [0] * total_perms
    for idx in window_perm:
        if idx >= 0:
            perm_count[idx] += 1
    return perm_count

def _find_critical_positions(word, n, window_perm, perm_count):
    """
    Find positions that are critical (contained in at least one unique perm window).
    Returns a boolean array where True means position is critical (cannot delete).
    """
    L = len(word)
    critical = [False] * L
    
    for s in range(len(window_perm)):
        if window_perm[s] >= 0 and perm_count[window_perm[s]] == 1:
            # This window is the only occurrence of its permutation
            for i in range(s, s + n):
                critical[i] = True
    
    return critical

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Fast deletion pass with incremental updates.
    A character at position i can be deleted if no valid permutation window
    that contains position i is the ONLY occurrence of that permutation.
    """
    word = list(word)
    L = len(word)
    
    # Initial computation
    window_perm = _compute_window_perms(word, n)
    perm_count = _compute_perm_counts(window_perm)
    
    total_deletions = 0
    max_rounds = 500
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        if len(word) < n:
            break
        
        # Find all critical positions
        critical = _find_critical_positions(word, n, window_perm, perm_count)
        
        # Safe positions are non-critical
        safe = [i for i, c in enumerate(critical) if not c]
        
        if not safe:
            break
        
        # Delete all safe positions at once (from right to left)
        for i in sorted(safe, reverse=True):
            if i < len(word):
                del word[i]
        
        total_deletions += len(safe)
        
        # Recompute
        window_perm = _compute_window_perms(word, n)
        perm_count = _compute_perm_counts(window_perm)
        
        # Early exit check
        if len(word) < n:
            break
    
    return word

def _targeted_insert_delete(word, n, time_limit, start_time):
    """
    Try inserting characters at strategic positions to enable more deletions.
    Focus on positions near critical windows.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    
    import random
    rng = random.Random(42)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current critical positions
        window_perm = _compute_window_perms(word, n)
        perm_count = _compute_perm_counts(window_perm)
        critical = _find_critical_positions(word, n, window_perm, perm_count)
        
        # Find a random critical position to try to "unlock"
        critical_positions = [i for i, c in enumerate(critical) if c]
        if not critical_positions:
            break
        
        target_pos = rng.choice(critical_positions)
        
        # Try inserting a character near the target position
        for offset in range(-3, 4):
            if time.time() - start_time > time_limit:
                break
            
            insert_pos = max(0, min(len(word), target_pos + offset))
            
            # Try each possible character
            for char in range(1, n + 1):
                if time.time() - start_time > time_limit:
                    break
                
                test_word = list(word)
                test_word.insert(insert_pos, char)
                
                # Quick deletion pass
                test_word = _fast_delete_pass(test_word, n, time_limit, start_time)
                
                # Check validity
                if _is_valid_superperm(test_word, n):
                    if len(test_word) < best_len:
                        best_word = list(test_word)
                        best_len = len(test_word)
    
    return best_word

def _adjacent_swap_optimization(word, n, time_limit, start_time):
    """
    Try adjacent swaps to potentially enable more deletions.
    Only try swaps that might reduce critical positions.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    
    import random
    rng = random.Random(123)
    
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        if len(word) < 2:
            break
        
        # Try a random adjacent swap
        i = rng.randint(0, len(word) - 2)
        
        # Check if the two characters are different
        if word[i] == word[i + 1]:
            continue
        
        # Swap
        word[i], word[i + 1] = word[i + 1], word[i]
        
        # Quick validity check using a set
        seen = set()
        for s in range(len(word) - n + 1):
            w = tuple(word[s:s + n])
            if len(set(w)) == n:
                seen.add(w)
        
        if len(seen) == total_perms:
            # Valid - try deletion pass
            test_word = _fast_delete_pass(list(word), n, time_limit, start_time)
            if _is_valid_superperm(test_word, n) and len(test_word) < best_len:
                best_word = list(test_word)
                best_len = len(test_word)
        else:
            # Revert
            word[i], word[i + 1] = word[i + 1], word[i]
    
    return best_word

def _multi_start_deletion(word, n, time_limit, start_time):
    """
    Try multiple random perturbations followed by deletion passes.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    
    import random
    rng = random.Random(456)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Create a perturbed version
        test_word = list(word)
        
        # Randomly delete a few non-critical positions first
        window_perm = _compute_window_perms(test_word, n)
        perm_count = _compute_perm_counts(window_perm)
        critical = _find_critical_positions(test_word, n, window_perm, perm_count)
        safe = [i for i, c in enumerate(critical) if not c]
        
        if safe:
            # Delete a random subset of safe positions
            num_delete = min(len(safe), rng.randint(1, 5))
            to_delete = rng.sample(safe, num_delete)
            for i in sorted(to_delete, reverse=True):
                if i < len(test_word):
                    del test_word[i]
        
        # Then try to insert characters to restore validity
        if not _is_valid_superperm(test_word, n):
            # Try inserting missing permutations
            seen = set()
            for s in range(len(test_word) - n + 1):
                w = tuple(test_word[s:s + n])
                if len(set(w)) == n:
                    seen.add(w)
            
            missing = [p for p in _PERM_LIST if p not in seen]
            
            if missing and len(missing) <= 10:
                # Try to insert a missing permutation with maximum overlap
                for perm in missing[:3]:
                    perm_list = list(perm)
                    
                    # Find best insertion point
                    best_insert_pos = -1
                    best_overlap = -1
                    
                    for pos in range(len(test_word) + 1):
                        # Check overlap with left
                        left_overlap = 0
                        for o in range(min(n, pos), 0, -1):
                            if test_word[pos - o:pos] == perm_list[:o]:
                                left_overlap = o
                                break
                        
                        # Check overlap with right
                        right_overlap = 0
                        for o in range(min(n, len(test_word) - pos), 0, -1):
                            if test_word[pos:pos + o] == perm_list[-o:]:
                                right_overlap = o
                                break
                        
                        total_overlap = left_overlap + right_overlap
                        if total_overlap > best_overlap:
                            best_overlap = total_overlap
                            best_insert_pos = pos
                    
                    if best_insert_pos >= 0:
                        # Insert with maximum overlap
                        new_test = list(test_word)
                        perm_to_insert = list(perm)
                        
                        # Determine what to actually insert
                        left_ov = 0
                        for o in range(min(n, best_insert_pos), 0, -1):
                            if new_test[best_insert_pos - o:best_insert_pos] == perm_to_insert[:o]:
                                left_ov = o
                                break
                        
                        right_ov = 0
                        for o in range(min(n, len(new_test) - best_insert_pos), 0, -1):
                            if new_test[best_insert_pos:best_insert_pos + o] == perm_to_insert[-o:]:
                                right_ov = o
                                break
                        
                        start = left_ov
                        end = len(perm_to_insert) - right_ov
                        if start < end:
                            new_test[best_insert_pos:best_insert_pos] = perm_to_insert[start:end]
                        
                        # Deletion pass
                        new_test = _fast_delete_pass(new_test, n, time_limit, start_time)
                        
                        if _is_valid_superperm(new_test, n) and len(new_test) < best_len:
                            best_word = list(new_test)
                            best_len = len(new_test)
        
        # Also try deletion on the perturbed word directly
        test_word = _fast_delete_pass(test_word, n, time_limit, start_time)
        if _is_valid_superperm(test_word, n) and len(test_word) < best_len:
            best_word = list(test_word)
            best_len = len(test_word)
    
    return best_word

def construct(n):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we use 140s to be safe
    time_limit = 140.0
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Phase 1: Fast batch deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Phase 2: Targeted insert-delete search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 20:
        word = _targeted_insert_delete(word, n, remaining_time, start_time)
    
    # Phase 3: Adjacent swap optimization
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15:
        word = _adjacent_swap_optimization(word, n, remaining_time, start_time)
    
    # Phase 4: Multi-start deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 20:
        word = _multi_start_deletion(word, n, remaining_time, start_time)
    
    # Phase 5: Final deletion pass
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    # Final safety check
    if time.time() - start_time > time_limit - 5:
        # Emergency: just validate and return
        if not _is_valid_superperm(word, n):
            word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
