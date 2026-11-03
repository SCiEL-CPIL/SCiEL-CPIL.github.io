"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START
def _build_recursive_superperm(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _get_all_permutations(n):
    """Generate all permutations of 1..n as tuples."""
    return set(permutations(range(1, n + 1)))

def _build_window_index(word, n):
    """
    Build a mapping from each length-n window (as tuple) to the list of
    start positions where it occurs.  Returns (window_to_positions, num_windows).
    """
    window_to_positions = {}
    num_windows = len(word) - n + 1
    for i in range(num_windows):
        w = tuple(word[i:i + n])
        if w in window_to_positions:
            window_to_positions[w].append(i)
        else:
            window_to_positions[w] = [i]
    return window_to_positions, num_windows

def _try_remove_with_index(word, pos, n, window_to_positions, required_perms):
    """
    Try to remove symbol at position pos using the window index.
    Returns (new_word, new_index, new_num_windows) if valid, else (None, None, None).
    
    Position pos is covered by windows starting at s in [max(0,pos-n+1), min(pos, len(word)-n)].
    For each such window w, if w occurs only at start s (i.e., len(window_to_positions[w])==1),
    then removing pos destroys w -> invalid.
    
    After removal, the new word has len-1 chars.  Windows in the new word that are
    unaffected (start positions far from pos) keep their occurrences.  Windows near pos
    need to be re-examined.  We rebuild the index locally for the affected region.
    """
    L = len(word)
    if pos < 0 or pos >= L:
        return None, None, None
    if L - 1 < n:
        return None, None, None
    
    # Determine which windows (by start index) include position pos
    s_min = max(0, pos - n + 1)
    s_max = min(pos, L - n)
    
    # Check: for each window covering pos, is it the ONLY occurrence?
    for s in range(s_min, s_max + 1):
        w = tuple(word[s:s + n])
        if len(window_to_positions[w]) == 1:
            # This permutation would be lost
            return None, None, None
    
    # Also: after removal, new windows are formed.  We need to ensure no
    # "new" window created by the splice is a permutation we haven't seen
    # before... actually no, we only need all required perms to still be present.
    # The old windows that survive are fine.  New windows at the splice point
    # are extra (they don't hurt validity, they might help or be duplicates).
    # The key check is: every required perm still has at least one occurrence.
    
    # Build new word
    new_word = word[:pos] + word[pos + 1:]
    new_L = len(new_word)
    
    # Rebuild window index for the new word (full rebuild is O(new_L * n) which is fine
    # for n=7, new_L ~ 5900)
    new_wtp = {}
    for i in range(new_L - n + 1):
        w = tuple(new_word[i:i + n])
        if w in new_wtp:
            new_wtp[w].append(i)
        else:
            new_wtp[w] = [i]
    
    # Verify all required permutations are present
    for p in required_perms:
        if p not in new_wtp:
            return None, None, None
    
    new_num_windows = new_L - n + 1
    return new_word, new_wtp, new_num_windows

def _optimize_word(word, n, time_limit):
    """
    Greedily try to remove symbols to shorten the word while keeping it valid.
    Uses a window index for O(n) per-removal validity checks (plus O(L*n) rebuild
    on successful removal).  Uses a time limit to avoid running too long.
    """
    start_time = time.time()
    required_perms = _get_all_permutations(n)
    
    current_word = word[:]
    removed_count = 0
    
    # Build initial window index
    wtp, num_w = _build_window_index(current_word, n)
    
    # Multiple passes: keep trying until no more removals possible or time runs out
    while True:
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        L = len(current_word)
        # Try removing each position in reverse order
        for pos in range(L - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            new_word, new_wtp, new_num_w = _try_remove_with_index(
                current_word, pos, n, wtp, required_perms)
            if new_word is not None:
                current_word = new_word
                wtp = new_wtp
                num_w = new_num_w
                removed_count += 1
                progress = True
                # Restart the loop since word changed
                break
        
        if not progress:
            break
    
    return current_word, removed_count

def construct(n):
    # Build the standard recursive superpermutation
    word = _build_recursive_superperm(n)
    
    # Time budget: leave 20 seconds for safety (evaluator kills at 150s)
    start_time = time.time()
    time_limit = 130.0  # seconds
    
    # Attempt to shorten the word
    if n <= 7:  # Only attempt optimization for reasonable n
        optimized_word, removed = _optimize_word(word, n, time_limit)
        if removed > 0:
            word = optimized_word
    
    # Convert to string
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
