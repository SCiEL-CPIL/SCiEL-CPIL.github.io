import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Step 1: Generate the classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of new_word
            overlap = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Step 2: Greedy Deletion / Local Search
    # We try to remove symbols to shorten the word while maintaining all permutations.
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
            if len(seen) == n ** n: # Max possible is n! but we check count
                pass
        # Actually, we need exactly n! permutations.
        # But checking len(seen) == factorial(n) is sufficient if we assume no duplicates?
        # No, a superpermutation must contain ALL permutations.
        # So we need len(seen) == math.factorial(n).
        import math
        return len(seen) == math.factorial(n)

    # Optimization: Pre-compute factorial
    import math
    target_perms = math.factorial(n)
    
    def is_valid_fast(w):
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == target_perms

    # Try to delete symbols
    # We do multiple passes.
    # To speed up, we can track the set of permutations.
    # But updating the set on deletion is tricky.
    # Given N ~ 6000, O(N) check is ~6000 ops.
    # N deletions * N checks = 36M ops.
    # Python might handle this in ~5-10s per pass.
    # We have 110s. So ~10-20 passes are possible.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order of deletion attempts to avoid local minima?
        # Or just left-to-right.
        # Let's try left-to-right.
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            
            # Try to delete word[i]
            # Create a copy to test
            # Optimization: Only check if deletion is potentially valid?
            # Hard to know without checking.
            
            # Fast check:
            # If we remove word[i], the windows affected are those containing i.
            # i.e., windows starting from max(0, i-n+1) to i.
            # We can compute the set of perms before, remove the affected ones,
            # and add the new ones formed by the shift.
            # This is complex. 
            # Let's stick to full re-validation but optimize the inner loop.
            
            # Actually, copying the list and checking is O(N).
            # Let's try it.
            
            new_word = word[:i] + word[i+1:]
            
            # Quick sanity: length must be at least n
            if len(new_word) < n:
                i += 1
                continue
            
            # Check validity
            seen = set()
            valid = True
            for j in range(len(new_word) - n + 1):
                seen.add(tuple(new_word[j:j+n]))
                if len(seen) > target_perms:
                    # Can't have more than n! unique perms of length n
                    # But we need exactly n!
                    pass
            
            if len(seen) == target_perms:
                word = new_word
                improved = True
                # Do not increment i, because the next element shifted into position i
                # might also be removable.
            else:
                i += 1
                
    return "".join(map(str, word))
