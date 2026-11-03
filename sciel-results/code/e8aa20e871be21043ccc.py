"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations
from collections import defaultdict

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

def _optimize_word_fast(word, n, time_limit):
    """
    Optimizes the word by removing redundant symbols using a sliding window count method.
    """
    start_time = time.time()
    required_perms = _get_all_permutations(n)
    
    # Initialize counts
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
        
    current_word = word[:]
    removed_count = 0
    
    # We iterate passes until no more removals or time limit
    # Try right-to-left first, then left-to-right if no progress
    directions = [range(len(current_word) - 1, -1, -1), range(len(current_word))]
    dir_idx = 0
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        progress = False
        # Get current direction
        if dir_idx == 0:
            pos_range = range(len(current_word) - 1, -1, -1)
        else:
            pos_range = range(len(current_word))
            
        for pos in pos_range:
            if time.time() - start_time > time_limit:
                break
            
            # Identify windows containing pos
            start_min = max(0, pos - n + 1)
            start_max = min(pos, len(current_word) - n)
            
            # Check if any window containing pos has count 1
            can_remove = True
            affected_windows = []
            for k in range(start_min, start_max + 1):
                w = tuple(current_word[k:k+n])
                if counts[w] == 1:
                    can_remove = False
                    break
                affected_windows.append(w)
            
            if not can_remove:
                continue
                
            # Perform removal
            new_word = current_word[:pos] + current_word[pos+1:]
            
            # Decrement affected
            for w in affected_windows:
                counts[w] -= 1
                
            # Calculate new windows
            # New content windows start at k where k < pos and k+n > pos
            # k in [max(0, pos-n+1), min(pos-1, len(new_word)-n)]
            new_affected = []
            max_new_start = len(new_word) - n
            for k in range(max(0, pos - n + 1), min(pos, max_new_start + 1)):
                if k >= 0:
                    w = tuple(new_word[k:k+n])
                    new_affected.append(w)
            
            # Increment new affected
            for w in new_affected:
                counts[w] += 1
                
            # Commit change
            current_word = new_word
            removed_count += 1
            progress = True
            # Break inner loop to restart with new indices and potentially switch direction
            break
            
        if not progress:
            # Switch direction
            dir_idx = (dir_idx + 1) % 2
            if dir_idx == 0 and removed_count > 0:
                # If we've made progress and are back at start, maybe we are stuck?
                # Or just continue?
                # If no progress in either direction, we are done.
                # We need a flag to know if we tried both.
                pass
            # Simple heuristic: if we tried both directions and made no progress in the last full cycle, stop.
            # To track this properly:
            if dir_idx == 0:
                 # We just finished direction 1 (LTR) and made no progress.
                 # We previously finished direction 0 (RTL) and made no progress?
                 # We don't track that easily.
                 # Let's just stop if no progress in LTR and we already did RTL?
                 # Actually, if we made progress in RTL, we restart RTL.
                 # If we made no progress in RTL, we try LTR.
                 # If we make no progress in LTR, we are stuck.
                 break
            else:
                 # We just finished direction 0 (RTL) and made no progress.
                 # Switch to LTR.
                 pass
                 
        # If we broke out of the for loop due to time, break while
        if time.time() - start_time > time_limit:
            break

    return current_word, removed_count

def construct(n):
    # Build the standard recursive superpermutation
    word = _build_recursive_superperm(n)
    
    # Time budget: leave 10 seconds for safety
    start_time = time.time()
    time_limit = 140.0  # seconds (Evaluator kills at 150s)
    
    # Attempt to shorten the word
    if n <= 7:  # Only attempt optimization for reasonable n
        optimized_word, removed = _optimize_word_fast(word, n, time_limit)
        if removed > 0:
            word = optimized_word
    
    # Convert to string
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
