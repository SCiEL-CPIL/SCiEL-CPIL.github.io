import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 105

    # Precompute all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    all_perms_set = set(all_perms)
    total_perms = len(all_perms)
    
    # Build a mapping from permutation to index for fast counting
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Helper to compute permutation counts using integer array
    def compute_counts(w):
        counts = [0] * total_perms
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            idx = perm_to_idx.get(t)
            if idx is not None:
                counts[idx] += 1
        return counts
    
    def is_valid_from_counts(counts):
        for c in counts:
            if c < 1:
                return False
        return True
    
    # Generate the naive superpermutation (recursive construction)
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    current_len = len(best_word)
    current_counts = compute_counts(best_word)
    
    if not is_valid_from_counts(current_counts):
        return "".join(map(str, best_word))
    
    # Main search loop with multiple phases
    while time.time() < deadline:
        found_improvement = False
        
        # Phase 1: Try to find redundant positions to remove
        # Use a faster check: for each position, see if any window containing it is unique
        for idx in range(len(best_word)):
            if len(best_word) <= n:
                break
            start_lo = max(0, idx - n + 1)
            start_hi = min(idx, len(best_word) - n)
            
            can_remove = True
            for s in range(start_lo, start_hi + 1):
                t = tuple(best_word[s:s+n])
                pi = perm_to_idx.get(t)
                if pi is not None and current_counts[pi] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                candidate = best_word[:]
                del candidate[idx]
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 2: Try swap moves (exchange two elements)
        if not found_improvement:
            for _ in range(500):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                i = random.randint(0, len(best_word) - 1)
                j = random.randint(0, len(best_word) - 1)
                if i == j:
                    continue
                candidate = best_word[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 3: Try insert-move (remove from one position, insert at another)
        if not found_improvement:
            for _ in range(1000):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                idx = random.randint(0, len(best_word) - 1)
                new_idx = random.randint(0, len(best_word) - 1)
                if idx == new_idx:
                    continue
                candidate = best_word[:]
                val = candidate.pop(idx)
                # Adjust new_idx if needed
                if new_idx > idx:
                    new_idx -= 1
                candidate.insert(new_idx, val)
                
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 4: If no improvement, try a larger perturbation
        if not found_improvement:
            if len(best_word) > n:
                # Try removing a random segment and reinserting
                seg_len = random.randint(1, min(3, len(best_word) - n))
                start_pos = random.randint(0, len(best_word) - seg_len)
                candidate = best_word[:]
                segment = candidate[start_pos:start_pos+seg_len]
                del candidate[start_pos:start_pos+seg_len]
                insert_pos = random.randint(0, len(candidate))
                candidate[insert_pos:insert_pos] = segment
                
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
    
    return "".join(map(str, best_word))
