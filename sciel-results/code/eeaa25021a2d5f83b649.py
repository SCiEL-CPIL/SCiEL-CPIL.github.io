import time
import random
import heapq
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate the naive superpermutation using standard construction
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
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Initialize permutation counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        perm_count[p] += 1
    
    # 3. Helper: check if position i is deletable
    def can_delete(i):
        L = len(word)
        if L <= n:
            return False
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        for j in range(start_j, end_j + 1):
            if perm_count[tuple(word[j:j + n])] == 1:
                return False
        return True
    
    # 4. Helper: delete position i and update counts
    def delete(i):
        L = len(word)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # Decrement old windows
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j + n])
            perm_count[p] -= 1
        
        # Commit deletion
        word.pop(i)
        
        # Increment new windows
        new_L = L - 1
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, new_L - n)
        
        for j in range(start_j_new, end_j_new + 1):
            new_w = tuple(word[j:j + n])
            perm_count[new_w] += 1
    
    # 5. Helper: check if swap (i, j) is valid
    def can_swap(i, j):
        if i == j or word[i] == word[j]:
            return False
        L = len(word)
        if L <= n:
            return False
        
        # Collect all affected window indices
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)
        
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)
        
        # Check if any affected window has count == 1
        for k in affected:
            if perm_count[tuple(word[k:k + n])] == 1:
                return False
        return True
    
    # 6. Helper: perform swap and update counts
    def do_swap(i, j):
        L = len(word)
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)
        
        old_vals = {k: tuple(word[k:k + n]) for k in affected}
        word[i], word[j] = word[j], word[i]
        
        for k in affected:
            old_p = old_vals[k]
            perm_count[old_p] -= 1
            new_p = tuple(word[k:k + n])
            perm_count[new_p] += 1
    
    # 7. Helper: undo swap
    def undo_swap(i, j):
        do_swap(i, j)  # Swapping back is the same operation
    
    # 8. Aggressive deletion pass: repeatedly delete any deletable position
    def greedy_delete():
        progress = True
        while progress:
            progress = False
            L = len(word)
            # Scan from left to right
            i = 0
            while i < L:
                if can_delete(i):
                    delete(i)
                    progress = True
                    L = len(word)
                    # Don't increment i; the next character shifted into position i
                else:
                    i += 1
                # Time check
                if time.time() > deadline - 5:
                    return
    
    # 9. Try a random swap followed by greedy deletion; keep if it reduces length
    def try_improve_swap():
        L_before = len(word)
        if L_before <= n + 1:
            return False
        
        # Pick two random distinct indices
        i = random.randint(0, L_before - 1)
        j = random.randint(0, L_before - 1)
        if i == j:
            return False
        
        if can_swap(i, j):
            do_swap(i, j)
            greedy_delete()
            if len(word) < L_before:
                return True
            else:
                # Revert the swap (note: word may have changed length due to deletions)
                # We need to find the current positions of the swapped elements
                # This is tricky; simpler to just not revert and accept the state
                # Actually, let's revert by swapping back the values at their new positions
                # But indices may have shifted. Let's just accept the current state
                # and rely on the fact that we only keep swaps that help.
                # For safety, let's just return False and leave the state as-is
                # (the deletions that happened are still valid)
                return False
        return False
    
    # 10. Main search loop
    # Phase 1: Initial aggressive deletion
    greedy_delete()
    
    # Phase 2: Alternating between swap-based improvement and deletion
    while time.time() < deadline - 3:
        # Try a swap-based improvement
        if try_improve_swap():
            continue
        
        # If no improvement from swap, try another random swap
        # Occasionally do a pure deletion pass to clean up
        if random.random() < 0.3:
            greedy_delete()
    
    # Final cleanup
    greedy_delete()
    
    return "".join(map(str, word))
