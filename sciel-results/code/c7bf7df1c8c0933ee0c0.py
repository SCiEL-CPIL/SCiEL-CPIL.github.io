import time
from itertools import permutations
import sys

def construct(n):
    # Safety margin: 150s limit, aim to finish by 145s
    deadline = time.time() + 145

    # 1. Generate standard superpermutation
    # Classic recursive construction: length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts array
    counts = [0] * num_perms
    L = len(word)
    
    # Count initial permutations
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Helper to check if all counts are >= 1
    # We can optimize this by tracking a 'missing' count, but simple check is O(5040) which is small
    # However, we only need to check if *any* count became 0 during the update.
    
    def try_delete(i):
        """
        Attempts to delete word[i]. 
        Returns True if successful (word and counts updated), False otherwise.
        """
        L_curr = len(word)
        if L_curr <= n:
            return False
        
        # Identify affected old windows
        # A window starting at k contains index i if k <= i <= k + n - 1
        # So k is in [i - n + 1, i]
        # Also k must be valid start: 0 <= k <= L_curr - n
        
        old_starts = []
        start_min = max(0, i - n + 1)
        start_max = min(i, L_curr - n)
        
        old_pids = []
        for k in range(start_min, start_max + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                old_pids.append(pid)
                # If count drops to 0, we must ensure it's restored by new windows
                # We don't fail immediately because new windows might restore it.
        
        # Identify affected new windows
        # After deleting index i, the word becomes word[:i] + word[i+1:]
        # The windows that change are those that span the deletion point.
        # In the new word (length L_curr - 1), let's call it W_new.
        # W_new[k] = word[k] if k < i, else word[k+1]
        
        # A window in W_new starting at k spans the deletion if:
        # It includes elements from before i and after i.
        # Specifically, the window W_new[k:k+n] contains the "seam" at i-1/i in old coords.
        # The elements involved are word[k...i-1] and word[i+1...k+n-1].
        # The window starts at k. The last element is at k+n-1.
        # The deletion is at i.
        # The window spans the deletion if k < i and k+n-1 >= i (in old coords, the gap is at i).
        # Actually, simpler: 
        # The windows in the NEW word that are different from the OLD word are those 
        # that include the position where the deletion happened.
        
        # Let's just compute the new windows in the range that could be affected.
        # Affected range in new word: starts k such that the window covers the seam.
        # The seam is between old index i-1 and i+1.
        # In new word, this is between index i-1 and i.
        # So new windows starting at k where k <= i-1 and k+n-1 >= i.
        # i.e., k in [i-n+1, i-1] (capped to valid range [0, L_curr - 1 - n])
        
        new_starts_min = max(0, i - n + 1)
        new_starts_max = min(i - 1, L_curr - 1 - n)
        
        new_pids = []
        for k in range(new_starts_min, new_starts_max + 1):
            # Construct the window from word with i removed
            # Part 1: word[k : i]
            # Part 2: word[i+1 : k+n]  (length n - (i-k))
            # Note: word[i+1 : k+n] in old word corresponds to the second part.
            # Let's verify indices.
            # New window is W_new[k : k+n].
            # If k < i:
            #   W_new[k...i-1] is word[k...i-1]
            #   W_new[i...k+n-1] is word[i+1...k+n]
            #   Total length: (i-k) + (k+n - (i+1) + 1) = i-k + n-i = n. Correct.
            
            w1 = word[k:i]
            w2 = word[i+1:k+n]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                new_pids.append(pid)

        # Check validity: no count should be 0
        # We only need to check the pids that were modified (decremented or incremented)
        # Actually, if a count was 1 and we decremented it, it becomes 0.
        # If it was restored by an increment, it becomes 1.
        # So we just need to check if any pid in old_pids has count 0 AND was not incremented in new_pids?
        # No, simply check if any count in the union of touched pids is 0.
        # Or simpler: check if any count in old_pids is 0?
        # If a pid in old_pids is 0, it means we lost that permutation.
        # It could be restored if it's also in new_pids.
        # So: if any pid in old_pids has count 0, we need to see if it's in new_pids.
        # Actually, just checking if any count in the *entire* array is 0 is O(5040).
        # Given the speed of the update, checking the full array is acceptable and safer.
        # 5040 checks is very fast in Python.
        
        is_valid = True
        # Optimization: Only check touched pids? 
        # If a pid was not touched, its count didn't change. If it was >0, it's still >0.
        # If it was 0, it would have been invalid before.
        # So we only need to check pids in old_pids (which were decremented).
        # If any pid in old_pids has count 0, it's invalid.
        # (New pids only increase counts, so they can't cause a 0).
        
        for pid in old_pids:
            if counts[pid] == 0:
                is_valid = False
                break

        if is_valid:
            # Commit deletion
            word.pop(i)
            return True
        else:
            # Revert counts
            for pid in old_pids:
                counts[pid] += 1
            for pid in new_pids:
                counts[pid] -= 1
            return False

    # Local Search Loop
    # We perform multiple passes. In each pass, we try to delete from left to right.
    # If a deletion is successful, the indices shift, so we should be careful.
    # Standard approach: 
    # i = 0
    # while i < len(word):
    #   if try_delete(i):
    #      # word got shorter, index i now points to the next element (shifted left)
    #      # So we do NOT increment i, we try to delete the new element at i again
    #      pass
    #   else:
    #      i += 1
      
    # However, try_delete modifies word. 
    # Let's implement a robust loop.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        L_curr = len(word)
        # Safety check: if L_curr is small, we might finish fast
        while i < L_curr:
            if time.time() > deadline - 5: # 5s safety buffer
                break
            
            if try_delete(i):
                improved = True
                L_curr -= 1
                # Do not increment i, as the element at i is now the one that was at i+1
            else:
                i += 1
        
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # For now, simple deletion is the primary strategy.
        # If we want to be more aggressive, we could try swapping, but deletion is the main gain.
        # Given the gap is small (8 chars), simple deletion should suffice if efficient enough.

    return "".join(map(str, word))

if __name__ == "__main__":
    # Quick self-test for n=4,5,6
    # print(construct(4))
    pass
