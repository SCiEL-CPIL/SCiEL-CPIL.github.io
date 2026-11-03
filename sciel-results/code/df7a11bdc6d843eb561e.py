import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Classic Recursive Construction
    # Length is 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break

    # For n <= 4, the recursive construction is optimal.
    # For n >= 5, we can try to shorten it.
    # The recursive construction for n=7 has length 5913.
    # The optimal is 5905. We can remove 8 characters.
    
    if n >= 5 and time.time() < deadline - 10.0:
        
        # Map permutations to IDs for O(1) count access
        # n! for n=7 is 5040. This is small.
        
        # Generate all permutations and map to ID
        # Using itertools.permutations
        import itertools
        all_perms = list(itertools.permutations(range(1, n+1)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        # Build initial counts
        counts = [0] * num_perms
        L = len(word)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
        
        # Verify validity
        if sum(1 for c in counts if c > 0) < num_perms:
             # Should not happen for recursive construction
             return "".join(map(str, word))

        # Local Search: Greedy Deletion
        # We scan from left to right.
        # If we delete a character, the word shrinks.
        # We restart the scan from max(0, i-1) after each successful deletion
        
        current_word = word
        
        def try_delete(i):
            """
            Attempts to delete current_word[i].
            Returns True if successful.
            Modifies current_word and counts if successful.
            """
            L = len(current_word)
            if L <= n:
                return False
            
            # 1. Identify old windows containing i
            # k in [max(0, i-n+1), min(i, L-n)]
            k_start_old = max(0, i - n + 1)
            k_end_old = min(i, L - n)
            
            old_pids = []
            for k in range(k_start_old, k_end_old + 1):
                p = tuple(current_word[k:k+n])
                if len(set(p)) == n:
                    pid = perm_to_id.get(p)
                    if pid is not None:
                        old_pids.append(pid)
                        counts[pid] -= 1
            
            # 2. Identify new windows formed
            # New word length L-1.
            # Junction is at i-1 and i in new word.
            # Windows in new word starting at k that span the junction:
            # k in [i-n+1, i-1]
            # Also k >= 0 and k+n-1 < L-1 => k < L-n
            k_start_new = max(0, i - n + 1)
            k_end_new = min(i - 1, L - n - 1)
            
            new_pids = []
            for k in range(k_start_new, k_end_new + 1):
                # New window is current_word[k:i] + current_word[i+1:k+n+1]
                w1 = current_word[k:i]
                w2 = current_word[i+1:k+n+1]
                p = tuple(w1 + w2)
                if len(set(p)) == n:
                    pid = perm_to_id.get(p)
                    if pid is not None:
                        new_pids.append(pid)
                        counts[pid] += 1
            
            # 3. Check validity
            # We need to ensure no count is 0.
            valid = True
            for pid in old_pids:
                if counts[pid] == 0:
                    valid = False
                    break
            if valid:
                for pid in new_pids:
                    if counts[pid] == 0: # Should not be 0 if we just incremented, but check if it was 0 before and is now 1? 
                        # Wait, if it was 0, it's not in the set of "covered" perms? 
                        # No, counts tracks occurrences. 
                        # If a perm had count 0, it's missing. 
                        # We only care about perms that *were* present (count > 0) becoming 0.
                        # New perms might have been 0, now 1. That's fine.
                        pass
            
            # Actually, we only need to check old_pids.
            # If an old perm drops to 0, it's invalid.
            # New perms becoming >0 is fine.
            
            if valid:
                # Commit deletion
                del current_word[i]
                return True
            else:
                # Revert
                for pid in old_pids:
                    counts[pid] += 1
                for pid in new_pids:
                    counts[pid] -= 1
                return False

        # Main deletion loop
        i = 0
        while i < len(current_word):
            if time.time() > deadline - 5.0:
                break
            
            if try_delete(i):
                i = max(0, i - 1)
            else:
                i += 1
                
            if time.time() > deadline - 5.0:
                break

        # Second pass
        if time.time() < deadline - 10.0:
            i = 0
            while i < len(current_word):
                if time.time() > deadline - 5.0:
                    break
                
                if try_delete(i):
                    i = max(0, i - 1)
                else:
                    i += 1
                    
                if time.time() > deadline - 5.0:
                    break

    return "".join(map(str, current_word))
