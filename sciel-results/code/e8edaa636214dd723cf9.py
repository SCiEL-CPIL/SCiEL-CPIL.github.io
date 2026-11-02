"""Superpermutation task: minimize the length of a word over {1..n} containing every n-permutation as a contiguous window."""
import time
import random
from itertools import permutations

def construct(n):
    t_start = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 120s kill

    if n <= 1:
        return "1" if n == 1 else ""

    # For n <= 4, use the classic recursive construction (optimal and fast)
    if n <= 4:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
            word = new
        return "".join(map(str, word))

    # For n >= 5: use the classic recursive construction as baseline.
    # This gives length sum_{i=1}^{n} i! = 5916 for n=7.
    # Then attempt local search / deletion to reduce length.
    
    def classic_construct(n):
        """Classic recursive superpermutation construction."""
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
            word = new
        return word

    sp = classic_construct(n)
    sp_len = len(sp)
    
    if time.time() - t_start > DEADLINE - 5:
        return "".join(map(str, sp))

    # --- Efficient deletion using precomputed coverage counts ---
    def efficient_deletions(word, n, max_time=30, direction="left_to_right"):
        """
        Find deletable positions efficiently.
        A position p is deletable iff for every window [j, j+n-1] containing p,
        that permutation appears in at least one other window NOT containing p.
        
        We precompute for each permutation how many times it appears in windows
        that contain a given position, and total occurrences.
        """
        t0 = time.time()
        word = list(word)
        
        while time.time() - t0 < max_time:
            L = len(word)
            if L <= n:
                break
            
            # Precompute: for each window start j, the permutation (if valid)
            # and for each position p, which windows contain it
            # perm_count[perm] = total number of windows with that perm
            # For efficiency, track: for each position p, the set of perms in windows containing p
            # and for each perm, count of windows NOT containing p
            
            # Build perm -> list of window start positions
            perm_windows = {}
            valid_windows = []  # (start, perm)
            for j in range(L - n + 1):
                w = tuple(word[j:j+n])
                if len(set(w)) == n:
                    valid_windows.append((j, w))
                    if w not in perm_windows:
                        perm_windows[w] = []
                    perm_windows[w].append(j)
            
            # Check coverage
            all_perms = set(permutations(range(1, n + 1)))
            if set(perm_windows.keys()) != all_perms:
                break
            
            # For each position, determine if it's deletable
            # Position p is contained in windows starting at j where j <= p <= j+n-1
            # i.e., j in [p-n+1, p] ∩ [0, L-n]
            # A position p is deletable iff for every valid window j containing p,
            # perm at j also appears at some window k NOT containing p.
            # NOT containing p means k > p or k + n - 1 < p, i.e., k > p or k < p - n + 1
            
            deleted = False
            positions = range(1, L - 1)
            if direction == "right_to_left":
                positions = range(L - 2, 0, -1)
            
            for pos in positions:
                if time.time() - t0 > max_time:
                    return word
                
                j_min = max(0, pos - n + 1)
                j_max = min(pos, L - n)
                
                can_delete = True
                for j in range(j_min, j_max + 1):
                    # Get the permutation at window j
                    # We need to check if this perm appears outside [j_min, j_max] range... 
                    # Actually, "not containing pos" means window start k such that k > pos OR k + n - 1 < pos
                    # k + n - 1 < pos => k < pos - n + 1 => k <= pos - n
                    # k > pos
                    w = tuple(word[j:j+n])
                    if len(set(w)) != n:
                        continue
                    
                    # Check if w appears at a window not containing pos
                    found_elsewhere = False
                    for k in perm_windows.get(w, []):
                        if k == j:
                            continue
                        # Window k does NOT contain pos if k > pos or k + n - 1 < pos
                        if k > pos or k + n - 1 < pos:
                            found_elsewhere = True
                            break
                    if not found_elsewhere:
                        can_delete = False
                        break
                
                if can_delete:
                    del word[pos]
                    deleted = True
                    break
        
            if not deleted:
                break
        
        return word

    # Apply deletion passes with different directions
    best_sp = list(sp)
    
    if time.time() - t_start < DEADLINE - 15:
        sp1 = efficient_deletions(sp, n, max_time=30, direction="left_to_right")
        if len(sp1) < len(best_sp):
            best_sp = sp1
    
    if time.time() - t_start < DEADLINE - 10:
        sp2 = efficient_deletions(best_sp, n, max_time=25, direction="right_to_left")
        if len(sp2) < len(best_sp):
            best_sp = sp2
    
    if time.time() - t_start < DEADLINE - 5:
        sp3 = efficient_deletions(best_sp, n, max_time=20, direction="left_to_right")
        if len(sp3) < len(best_sp):
            best_sp = sp3

    # --- Reinsertion-deletion cycles to escape local optima ---
    if time.time() - t_start < DEADLINE - 5:
        for _ in range(5):
            if time.time() - t_start > DEADLINE - 5:
                break
            # Randomly insert a symbol, then try to delete it back (possibly elsewhere)
            candidate = list(best_sp)
            L = len(candidate)
            if L < n:
                break
            ins_pos = random.randint(1, L - 1)
            ins_val = random.randint(1, n)
            candidate.insert(ins_pos, ins_val)
            
            # Now try to delete the inserted symbol or others
            result = efficient_deletions(candidate, n, max_time=10, direction="left_to_right")
            if len(result) < len(best_sp):
                best_sp = result

    # Final deadline check
    if time.time() - t_start > DEADLINE - 2:
        return "".join(map(str, best_sp))

    return "".join(map(str, best_sp))
