import time
import random

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # 1. Generate the naive superpermutation
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
            # Find max overlap
            max_overlap = 0
            # Max possible overlap is len(new) and len(block)
            # But we only care about suffix of new matching prefix of block
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # 2. Greedy Deletion Search
    # We want to remove characters while keeping all permutations.
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            # Check if window is a permutation
            # Optimization: check if all elements are distinct and in 1..n
            # Using tuple is standard
            p = tuple(w[i:i+n])
            if p in seen:
                continue
            # Check if it's a valid permutation of 1..n
            # Since w is derived from permutations, it likely is, 
            # but deletions might create invalid windows? 
            # No, we only check if it CONTAINS all valid permutations.
            # Invalid windows are ignored.
            # So we just need to ensure all 5040 valid perms are in 'seen'.
            seen.add(p)
        
        # Check if seen contains all permutations of 1..n
        # This is expensive. 
        # Better: maintain a count of found permutations.
        pass

    # Faster validity check:
    # Pre-generate all permutations? 5040 is small.
    import itertools
    all_perms = set(map(tuple, itertools.permutations(range(1, n+1))))
    
    def is_valid_fast(w):
        if len(w) < n:
            return False
        found = set()
        # Early exit if we can't find all
        # We just add to found.
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            found.add(p)
            # Optimization: if len(found) == len(all_perms), we can return True?
            # No, we need to check if found is a SUPERSET of all_perms?
            # No, w only contains valid perms if we ensure it.
            # But deletions might create invalid tuples (e.g. 1,1,2...).
            # So we should intersect with all_perms.
            if p in all_perms:
                found.add(p)
                if len(found) == len(all_perms):
                    return True
        return len(found) == len(all_perms)

    # The check `len(found) == len(all_perms)` is sufficient if we only add valid perms.
    
    # Greedy Loop
    current_word = word
    # Try to delete characters.
    # We can try positions in random order or sequential.
    # Sequential with restart is better for finding dense deletions.
    
    # To speed up, we can check deletions in batches?
    # No, just loop.
    
    # We need to be careful with time.
    # is_valid_fast takes O(L * n) roughly.
    # L ~ 6000. n=7.
    # 42000 ops.
    # 1000 checks = 42M ops. ~5s in Python?
    # So we can do ~2000 checks in 10s.
    # We have 110s. So ~20,000 checks.
    
    # Strategy:
    # While time < deadline:
    #   Try to delete a random character.
    #   If valid, keep it.
    #   If not, try another.
    #   To avoid getting stuck, we can also try to "shake" the string?
    #   But deletions only reduce length.
    
    # A better strategy:
    # Iterate through indices. For each index, try to delete.
    # If successful, restart iteration from 0.
    # If we pass the end without deletion, stop?
    # Or randomize the order of indices.
    
    random.seed(42)
    
    # Pre-calculate all_perms
    all_perms = set(map(tuple, itertools.permutations(range(1, n+1))))
    target_count = len(all_perms)
    
    def check_validity(w):
        found = set()
        # Use local vars for speed
        ap = all_perms
        n_local = n
        L = len(w)
        for i in range(L - n_local + 1):
            p = tuple(w[i:i+n_local])
            if p in ap:
                found.add(p)
                if len(found) == target_count:
                    return True
        return len(found) == target_count

    # Initial check
    if not check_validity(current_word):
        return "".join(map(str, current_word))
        
    # Deletion Search
    # We will try to delete characters.
    # We can prioritize positions that are "redundant".
    # A character is redundant if its removal doesn't break any unique permutation occurrence.
    # But checking that is complex.
    
    # Simple Random Deletion with Backtracking?
    # No, just accept valid deletions.
    
    # To avoid infinite loops or slow progress, we can limit retries per position.
    
    # Let's try a "scan" approach.
    # We scan from left to right.
    # If we can delete w[i], we do it.
    # Then we restart the scan from i-1 (or 0).
    # This finds a local minimum for this specific order.
    # We can repeat this with different random orders or restarts.
    
    best_word = current_word
    best_len = len(current_word)
    
    # We can try multiple passes with different random shuffles of indices?
    # Or just one long pass?
    
    # Let's try a loop that attempts deletions.
    # We'll use a random order of indices to avoid bias.
    
    while time.time() < deadline:
        # Create a list of indices to try
        # Trying all indices every time is O(L^2) checks.
        # L=6000. 36M checks. Too slow.
        
        # Instead, try a subset of indices per "epoch".
        # Or just pick a random index.
        
        idx = random.randint(0, len(current_word) - 1)
        
        # Try to delete current_word[idx]
        new_word = current_word[:idx] + current_word[idx+1:]
        
        if check_validity(new_word):
            current_word = new_word
            if len(current_word) < best_len:
                best_len = len(current_word)
                best_word = current_word
        else:
            # Optional: try swapping idx and idx+1 to unstick?
            # For now, just move on.
            pass
            
    return "".join(map(str, best_word))
