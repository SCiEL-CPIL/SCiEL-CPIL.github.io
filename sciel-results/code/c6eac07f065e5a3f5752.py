import random
import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 115  # 115 seconds to be safe
    
    def time_left():
        return deadline - time.time()
    
    # For n=7, use efficient permutation handling
    if n <= 7:
        all_perms = list(permutations(range(1, n+1)))
    else:
        # For larger n, use a more efficient approach
        all_perms = []
        def generate_perms(prefix, remaining):
            if not remaining:
                all_perms.append(prefix)
            else:
                for i in range(len(remaining)):
                    generate_perms(prefix + (remaining[i],), 
                                  remaining[:i] + remaining[i+1:])
        generate_perms((), tuple(range(1, n+1)))
    
    # Greedy construction: build word by adding permutations with maximum overlap
    # Start with first permutation
    word = list(all_perms[0])
    remaining = set(all_perms[1:])
    used = {all_perms[0]}
    
    while remaining and time_left() > 0.5:
        # Find which permutation gives maximum overlap with current suffix
        suffix_len = min(n, len(word))
        current_suffix = tuple(word[-suffix_len:])
        
        best_perm = None
        best_overlap = 0
        
        for perm in list(remaining):
            # Find maximum overlap: how much of perm matches end of word
            for overlap in range(n-1, 0, -1):
                if tuple(word[-overlap:]) == perm[:overlap]:
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_perm = perm
                    break
        
        if best_perm is None:
            # No overlap found, pick a random remaining permutation
            best_perm = random.choice(list(remaining))
            best_overlap = 0
        
        # Add the new part of the permutation
        if best_overlap > 0:
            word.extend(best_perm[best_overlap:])
        else:
            word.extend(best_perm)
        
        remaining.discard(best_perm)
        used.add(best_perm)
        
        # Check if we've added all permutations
        if not remaining:
            break
    
    # Ensure all permutations are in the word (verify and add missing ones if needed)
    def encode_window(w, pos):
        return tuple(w[pos:pos+n])
    
    # Check which permutations are missing
    found = set()
    for j in range(len(word) - n + 1):
        w = encode_window(word, j)
        if len(set(w)) == n:
            found.add(w)
    
    missing = set(all_perms) - found
    if missing and time_left() > 1.0:
        # Add missing permutations one by one
        for perm in list(missing):
            if time_left() < 1.0:
                break
            # Try to find maximum overlap
            best_overlap = 0
            for overlap in range(n-1, 0, -1):
                if len(word) >= overlap and tuple(word[-overlap:]) == perm[:overlap]:
                    best_overlap = overlap
                    break
            if best_overlap > 0:
                word.extend(perm[best_overlap:])
            else:
                word.extend(perm)
            missing.discard(perm)
    
    # Convert to list of ints
    word = list(word)
    
    # Local search: iteratively delete positions that don't break coverage
    def get_perm_counts(w):
        counts = {}
        for j in range(len(w) - n + 1):
            p = tuple(w[j:j+n])
            if len(set(p)) == n:
                counts[p] = counts.get(p, 0) + 1
        return counts
    
    def can_delete_at(w, idx, perm_counts):
        # Check if deleting position idx would remove the last occurrence of any permutation
        # A permutation at position j covers positions j to j+n-1
        # Position idx is covered by permutations starting at max(0, idx-n+1) to min(idx, len(w)-n)
        
        affected = set()
        start_min = max(0, idx - n + 1)
        start_max = min(idx, len(w) - n)
        
        for j in range(start_min, start_max + 1):
            p = tuple(w[j:j+n])
            if len(set(p)) == n and perm_counts.get(p, 0) == 1:
                affected.add(p)
        
        return len(affected) == 0
    
    # Iterative deletion
    iteration = 0
    while time_left() > 2.0:
        iteration += 1
        
        perm_counts = get_perm_counts(word)
        
        # Find all deletable positions
        candidates = []
        for idx in range(len(word)):
            if can_delete_at(word, idx, perm_counts):
                candidates.append(idx)
        
        if not candidates:
            break  # Local minimum reached
        
        # Delete one random candidate
        idx_to_delete = random.choice(candidates)
        word.pop(idx_to_delete)
        
        # Occasionally do a more aggressive pass
        if iteration % 100 == 0 and time_left() > 5.0:
            # Try multiple deletions
            perm_counts = get_perm_counts(word)
            for _ in range(10):
                if time_left() < 3.0:
                    break
                candidates = [i for i in range(len(word)) if can_delete_at(word, i, perm_counts)]
                if not candidates:
                    break
                idx = random.choice(candidates)
                word.pop(idx)
                perm_counts = get_perm_counts(word)
    
    # Final verification and return
    # Return as string of digits
    return "".join(map(str, word))
