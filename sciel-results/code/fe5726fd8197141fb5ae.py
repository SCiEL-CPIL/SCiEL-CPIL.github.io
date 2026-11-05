import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Generate standard recursive superpermutation
    # This creates a word of length sum_{k=1}^n k!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique (m-1)-permutations in order of appearance
        # Note: For m=2, we look at 1-perms. For m=3, 2-perms, etc.
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # A valid (m-1)-permutation must have distinct elements
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Block is P, m, P
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    S = list(word)
    best_S = S[:]
    best_len = len(S)
    
    # Bit-packing helper for fast window extraction
    # n is at most 7, so 3 bits per symbol is enough (0-7)
    # We map symbol 1->1, 2->2, ..., 7->7. 0 is unused.
    # Window of n symbols = 3*n bits.
    
    def pack_word(S):
        """Pack list of ints into a list of bit-ints? No, just keep list for mutation, 
        but use a rolling hash/bitmask for fast window checks?
        Actually, for n=7, tuple is okay if we optimize the loop.
        But let's try a different approach: 
        Since we are doing many checks, let's precompute a 'signature' or just use tuples but optimize the loop.
        
        Let's stick to tuples but make the removal pass very efficient.
        """
        return S

    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def is_valid(S, n):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        total = 1
        for k in range(2, n+1):
            total *= k
        return len(seen) == total

    def local_search(S, n, deadline):
        """
        Optimized local search using deletion passes.
        We alternate forward and backward passes.
        """
        counts = build_counts(S, n)
        
        # To speed up, we can use a list of counts and a mapping from tuple to index?
        # No, dictionary is fine for 5040 keys.
        
        def remove_pass(S, counts, n, deadline, forward=False):
            removed_any = False
            L = len(S)
            
            if forward:
                i = 0
                while i < L:
                    if time.time() > deadline:
                        break
                    # Windows affected by removing S[i]
                    # Old windows: [i-n+1, i]
                    # New windows: [i-n+1, i-1] in the new string (shifted)
                    
                    start_old = max(0, i - n + 1)
                    end_old = min(i, L - n)
                    
                    valid = True
                    old_perms = []
                    for j in range(start_old, end_old + 1):
                        p = tuple(S[j:j+n])
                        if counts.get(p, 0) == 1:
                            valid = False
                            break
                        old_perms.append(p)
                    
                    if not valid:
                        i += 1
                        continue
                    
                    # If valid, remove
                    for p in old_perms:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    
                    # Add new windows
                    # New length is L-1
                    # New windows straddling the gap:
                    # Indices in new string: k such that k < i <= k+n-1
                    # i.e., i-n+1 <= k <= i-1
                    start_new = max(0, i - n + 1)
                    end_new = min(i - 1, L - 1 - n)
                    
                    for k in range(start_new, end_new + 1):
                        # Construct window from S
                        # Indices in S:
                        # if idx < i: S[idx]
                        # if idx >= i: S[idx+1]
                        window_chars = []
                        for idx in range(k, k+n):
                            s_idx = idx if idx < i else idx + 1
                            window_chars.append(S[s_idx])
                        p_new = tuple(window_chars)
                        counts[p_new] = counts.get(p_new, 0) + 1
                    
                    S.pop(i)
                    L -= 1
                    removed_any = True
                    # Do not increment i, check same position again
            else:
                i = L - 1
                while i >= 0:
                    if time.time() > deadline:
                        break
                    start_old = max(0, i - n + 1)
                    end_old = min(i, L - n)
                    
                    valid = True
                    old_perms = []
                    for j in range(start_old, end_old + 1):
                        p = tuple(S[j:j+n])
                        if counts.get(p, 0) == 1:
                            valid = False
                            break
                        old_perms.append(p)
                    
                    if not valid:
                        i -= 1
                        continue
                    
                    for p in old_perms:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    
                    start_new = max(0, i - n + 1)
                    end_new = min(i - 1, L - 1 - n)
                    
                    for k in range(start_new, end_new + 1):
                        window_chars = []
                        for idx in range(k, k+n):
                            s_idx = idx if idx < i else idx + 1
                            window_chars.append(S[s_idx])
                        p_new = tuple(window_chars)
                        counts[p_new] = counts.get(p_new, 0) + 1
                    
                    S.pop(i)
                    L -= 1
                    removed_any = True
                    # Do not decrement i
            
            return S, counts, removed_any

        # Run alternating passes
        # We need to be careful: if we remove a char, the counts change.
        # The current implementation updates counts correctly.
        
        while time.time() < deadline:
            S, counts, r1 = remove_pass(S, counts, n, deadline, forward=False)
            if not r1:
                # Try forward
                S, counts, r2 = remove_pass(S, counts, n, deadline, forward=True)
                if not r2:
                    # Stuck
                    break
            # If r1 was true, we continue loop to try more backward passes?
            # Or alternate? 
            # Usually, if backward pass removes something, another backward pass might find more.
            # Let's just loop until no changes in a full cycle (B then F).
            
        return S

    # ILS Loop
    current_S = best_S[:]
    
    # Initial local search
    current_S = local_search(current_S, n, deadline)
    if len(current_S) < best_len:
        best_S = current_S
        best_len = len(best_S)

    # Iterated Local Search
    # If we are stuck, we need to perturb.
    # Perturbation: Insert random chars? Or Swap?
    # Inserting chars increases length, which is bad, but might allow removal of others later?
    # Actually, the goal is to remove chars.
    # If we are stuck, it means no single char can be removed.
    # A swap of two chars might enable a removal.
    
    stuck_counter = 0
    last_best_len = best_len
    
    while time.time() < deadline:
        # Perturb current_S
        S_perturbed = current_S[:]
        
        # Try a few random swaps
        # A swap of S[i] and S[j] (i != j)
        # We check if the swap is valid (doesn't create a new unique window that was previously unique? 
        # No, swap changes windows. We must ensure no window count drops to 0.
        # This is expensive to check directly.
        # Alternative: Just do the swap, run local search, and if it fails (length doesn't improve or validity breaks), revert.
        # But validity check is expensive.
        
        # Better perturbation: 
        # Take a random substring and reverse it? Or shift it?
        
        # Let's try: Pick two random positions i, j. Swap S[i] and S[j].
        # Then run local search.
        # If the result is valid and shorter than current_S, accept.
        # If not, revert.
        
        if len(S_perturbed) > 1:
            i = random.randint(0, len(S_perturbed) - 1)
            j = random.randint(0, len(S_perturbed) - 1)
            if i != j:
                S_perturbed[i], S_perturbed[j] = S_perturbed[j], S_perturbed[i]
        
        # Run local search on perturbed
        S_ls = local_search(S_perturbed, n, deadline)
        
        # Check validity of S_ls
        if is_valid(S_ls, n):
            if len(S_ls) < best_len:
                best_S = S_ls
                best_len = len(S_ls)
                current_S = S_ls[:]
                stuck_counter = 0
            elif len(S_ls) < len(current_S):
                # Accept if it's shorter than current, even if not best
                current_S = S_ls[:]
                stuck_counter = 0
            else:
                stuck_counter += 1
                # If stuck for too long, try a stronger perturbation
                if stuck_counter > 5:
                    # Insert a random char to break symmetry?
                    # This increases length, but might allow a different deletion path.
                    pos = random.randint(0, len(current_S))
                    current_S.insert(pos, random.randint(1, n))
                    # Run LS on this expanded string
                    S_ls = local_search(current_S, n, deadline)
                    if is_valid(S_ls, n) and len(S_ls) < best_len:
                        best_S = S_ls
                        best_len = len(S_ls)
                    current_S = S_ls if is_valid(S_ls, n) else current_S
                    stuck_counter = 0
        else:
            # Invalid, revert
            stuck_counter += 1

        # Safety check
        if time.time() > deadline - 5:
            break

    # Final Polish
    if time.time() < deadline - 2:
        best_S = local_search(best_S, n, deadline)
        if is_valid(best_S, n):
            best_len = len(best_S)

    # Final Validity Check
    if not is_valid(best_S, n):
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
