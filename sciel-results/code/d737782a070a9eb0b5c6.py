import time
import random
import sys
from itertools import permutations

# Precompute permutation indices for n=7?
# Since n is passed to construct, we should handle it dynamically, 
# but the task says "given as a string of the digits 1..n or a list of integers; the evaluator passes this task's n".
# The task is specifically for n=7 based on the description "Shortest superpermutation on seven symbols".
# But the interface is construct(n).
# I will assume n=7 is the primary case, but write generic code for n <= 7 or so.

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    if n > 7:
        # Fallback to recursive for larger n, though score will be poor
        word = "1"
        for m in range(2, n + 1):
            perms = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = word[i:i + m - 1]
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            new_word = ""
            for p in perms:
                block = p + str(m) + p
                max_overlap = 0
                limit = min(len(new_word), len(block))
                for o in range(limit, 0, -1):
                    if new_word[-o:] == block[:o]:
                        max_overlap = o
                        break
                new_word += block[max_overlap:]
            word = new_word
        return word

    # n is likely 7
    # 1. Generate Recursive Superpermutation
    word = "1"
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word
        
    arr = [int(c) for c in word]
    L = len(arr)
    
    # Precompute all permutations of 1..n and map to index
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Helper to get perm index
    def get_perm_idx(arr, start):
        if start < 0 or start + n > len(arr):
            return -1
        p = tuple(arr[start:start+n])
        return perm_to_idx.get(p, -1)

    # Initialize state
    # perm_count[i] = count of permutation i
    perm_count = [0] * num_perms
    # perm_idx[i] = index of perm starting at i, or -1
    perm_idx = [-1] * L
    
    # Initial scan
    for i in range(L - n + 1):
        p_idx = get_perm_idx(arr, i)
        if p_idx != -1:
            perm_idx[i] = p_idx
            perm_count[p_idx] += 1
            
    # Verify all perms are covered
    # (Should be true for recursive construction)
    
    # Criticality: 
    # pos_critical[i] = number of unique perms that cover position i
    pos_critical = [0] * L
    
    def update_criticality(arr, perm_idx, perm_count, pos_critical, n, L):
        # Reset pos_critical
        pos_critical = [0] * L
        for i in range(L - n + 1):
            p_idx = perm_idx[i]
            if p_idx != -1 and perm_count[p_idx] == 1:
                for k in range(n):
                    pos_critical[i+k] += 1
        return pos_critical

    pos_critical = update_criticality(arr, perm_idx, perm_count, pos_critical, n, L)
    
    best_len = L
    best_arr = arr[:]
    
    def try_deletions(arr, perm_idx, perm_count, pos_critical, L, n, num_perms, time_limit):
        # Local search: greedily delete non-critical chars
        # We need to update state carefully.
        
        # To speed up, we can just scan and delete if possible.
        # If we delete at i, we affect windows [i-n+1, i].
        # We need to update perm_idx and perm_count for these windows.
        # Then re-evaluate criticality for affected positions.
        
        # This is complex to do incrementally in Python without bugs.
        # Given L ~ 5900 and only ~8 deletions needed, we can afford a slightly slower check.
        
        # Strategy: 
        # 1. Find all i where pos_critical[i] == 0.
        # 2. Pick one, delete it.
        # 3. Update local state.
        # 4. Repeat.
        
        # To avoid O(L) criticality update on every deletion, 
        # we can just re-calculate criticality from scratch? 
        # O(L) per deletion. 8 deletions * 5900 = 47k ops. Very fast.
        
        while time.time() < time_limit:
            # Find deletable index
            del_idx = -1
            for i in range(L):
                if pos_critical[i] == 0:
                    del_idx = i
                    break
            
            if del_idx == -1:
                return arr, L # Stuck
            
            # Delete del_idx
            # 1. Update perm_count and perm_idx for windows affected by deletion
            # Old windows affected: j in [del_idx - n + 1, del_idx]
            for j in range(max(0, del_idx - n + 1), min(del_idx, L - n) + 1):
                p_idx = perm_idx[j]
                if p_idx != -1:
                    perm_count[p_idx] -= 1
                    perm_idx[j] = -1 # Mark as invalid temporarily
            
            # 2. Pop element
            arr.pop(del_idx)
            L -= 1
            
            # 3. Update perm_idx and perm_count for new windows
            # New windows affected: j in [del_idx - n + 1, del_idx - 1]
            # Note: indices >= del_idx are shifted left by 1.
            # perm_idx array needs to be updated for the shift too?
            # It's easier to just re-calculate perm_idx for the whole array? No, O(L).
            # But we only need to update the affected region.
            # However, the shift affects perm_idx indices for j >= del_idx.
            # perm_idx[j] was for window starting at j. Now window starting at j starts at old j+1.
            # So perm_idx[j] should become old perm_idx[j+1].
            
            # Let's just re-calculate perm_idx for the entire array to be safe and simple.
            # L is ~5900. O(L) is fast.
            for i in range(L - n + 1):
                perm_idx[i] = get_perm_idx(arr, i)
            # Reset perm_count
            for i in range(num_perms):
                perm_count[i] = 0
            for i in range(L - n + 1):
                if perm_idx[i] != -1:
                    perm_count[perm_idx[i]] += 1
            
            # 4. Re-calculate criticality
            pos_critical = [0] * L
            for i in range(L - n + 1):
                if perm_idx[i] != -1 and perm_count[perm_idx[i]] == 1:
                    for k in range(n):
                        pos_critical[i+k] += 1
            
            if L < best_len:
                best_len = L
                best_arr = arr[:]
                
        return best_arr, best_len

    # Main loop with restarts/perturbations
    # Since greedy might get stuck, we try multiple times with random perturbations if stuck.
    
    # For n=7, the gap is small. A single greedy pass might work if we pick the "right" deletion.
    # But there might be multiple choices.
    
    # Let's try a few random restarts if the first one doesn't reach 5905.
    target_len = 5905
    
    attempts = 0
    max_attempts = 10 # Should be enough given speed
    
    while time.time() < deadline - 5:
        # Reset state to original
        arr = [int(c) for c in word]
        L = len(arr)
        perm_idx = [-1] * L
        perm_count = [0] * num_perms
        for i in range(L - n + 1):
            p_idx = get_perm_idx(arr, i)
            if p_idx != -1:
                perm_idx[i] = p_idx
                perm_count[p_idx] += 1
        pos_critical = [0] * L
        for i in range(L - n + 1):
            if perm_idx[i] != -1 and perm_count[perm_idx[i]] == 1:
                for k in range(n):
                    pos_critical[i+k] += 1
        
        # If not first attempt, apply random perturbation
        if attempts > 0:
            # Swap two random elements?
            # Or insert a random element?
            # Swapping might break superpermutation property temporarily, but we only delete if safe.
            # Actually, we start from a valid superpermutation.
            # Perturbation: Pick a random position, try to delete it? No, that's what we're doing.
            # Maybe shuffle the order of deletion?
            # Or, add a random character and try to delete it back?
            
            # Simple perturbation: Swap two adjacent characters?
            # This might create new unique perms or remove existing ones.
            # Let's just try a different random seed for deletion choice if we had multiple choices.
            # In my `try_deletions`, I pick the first one.
            # Let's modify `try_deletions` to pick randomly among available deletions.
            
            pass # Handled in try_deletions modification below
        
        # Run greedy deletion
        # We need a version that picks randomly
        current_arr, current_L = arr, L
        current_perm_idx = perm_idx
        current_perm_count = perm_count
        current_pos_critical = pos_critical
        
        while time.time() < deadline - 10:
            # Find all deletable indices
            candidates = [i for i in range(current_L) if current_pos_critical[i] == 0]
            
            if not candidates:
                break
            
            # Pick one
            # For first attempt, pick first. For others, pick random.
            if attempts == 0:
                del_idx = candidates[0]
            else:
                del_idx = random.choice(candidates)
                
            # Delete del_idx
            # Update logic as before
            for j in range(max(0, del_idx - n + 1), min(del_idx, current_L - n) + 1):
                p_idx = current_perm_idx[j]
                if p_idx != -1:
                    current_perm_count[p_idx] -= 1
                    current_perm_idx[j] = -1
            
            current_arr.pop(del_idx)
            current_L -= 1
            
            # Recalculate perm_idx and perm_count (O(L))
            # Resize perm_idx if needed? It's a list, just overwrite.
            # We only care about indices 0..current_L-n
            for i in range(current_L - n + 1):
                current_perm_idx[i] = get_perm_idx(current_arr, i)
            for i in range(num_perms):
                current_perm_count[i] = 0
            for i in range(current_L - n + 1):
                if current_perm_idx[i] != -1:
                    current_perm_count[current_perm_idx[i]] += 1
            
            # Recalculate criticality
            current_pos_critical = [0] * current_L
            for i in range(current_L - n + 1):
                if current_perm_idx[i] != -1 and current_perm_count[current_perm_idx[i]] == 1:
                    for k in range(n):
                        current_pos_critical[i+k] += 1
                        
            if current_L < best_len:
                best_len = current_L
                best_arr = current_arr[:]
                
        attempts += 1
        if best_len == target_len:
            break
            
    return ''.join(str(x) for x in best_arr)
