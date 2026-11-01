import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Hardcoded for small n
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    if n != 7:
        # Fallback for other n (though task implies n=7)
        # Use recursive construction
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

    # n = 7 specific logic
    # 1. Generate recursive superpermutation
    word_str = "1"
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word_str) - (m - 1) + 1):
            window = word_str[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word_str = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word_str), len(block))
            for o in range(limit, 0, -1):
                if new_word_str[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word_str += block[max_overlap:]
        word_str = new_word_str
    
    arr = [int(c) for c in word_str]
    L = len(arr)
    
    # Precompute permutation mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Helper to get perm index for a window in arr
    def get_perm_idx(arr, start):
        if start < 0 or start + n > len(arr):
            return -1
        p = tuple(arr[start:start+n])
        return perm_to_idx.get(p, -1)

    # State initialization
    perm_idx = [-1] * L
    perm_count = [0] * num_perms
    
    for i in range(L - n + 1):
        p_idx = get_perm_idx(arr, i)
        if p_idx != -1:
            perm_idx[i] = p_idx
            perm_count[p_idx] += 1
            
    # Criticality: pos_critical[i] = number of unique perms covering i
    pos_critical = [0] * L
    for i in range(L - n + 1):
        p_idx = perm_idx[i]
        if p_idx != -1 and perm_count[p_idx] == 1:
            for k in range(n):
                pos_critical[i+k] += 1

    best_len = L
    best_arr = arr[:]
    target_len = 5905

    attempts = 0
    max_attempts = 100000 # Increase attempts since each is faster
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        # Reset state
        curr_arr = arr[:]
        curr_L = L
        curr_perm_idx = perm_idx[:]
        curr_perm_count = perm_count[:]
        curr_pos_critical = pos_critical[:]
        
        while time.time() < deadline - 5:
            # Find deletable indices
            candidates = []
            for i in range(curr_L):
                if curr_pos_critical[i] == 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Pick random candidate
            del_idx = random.choice(candidates)
            
            # Perform incremental deletion
            # 1. Identify windows affected: j in [del_idx - n + 1, del_idx]
            # These windows contain the element at del_idx.
            
            # We need to update perm_count for these windows.
            # And update pos_critical for positions in these windows.
            
            # Store the old perm_idx for these windows to handle the "count drops to 1" case
            affected_windows = []
            for j in range(max(0, del_idx - n + 1), min(del_idx, curr_L - n) + 1):
                p_idx = curr_perm_idx[j]
                if p_idx != -1:
                    affected_windows.append((j, p_idx))
            
            # Decrement counts and track which perms become unique (count 1)
            # and which were already unique (count 1 -> 0, but we checked criticality so this shouldn't happen for valid deletion)
            # Wait, if pos_critical[del_idx] == 0, then for all p covering del_idx, count[p] > 1.
            # So count[p] will become >= 1.
            
            new_unique_perms = []
            for j, p_idx in affected_windows:
                curr_perm_count[p_idx] -= 1
                if curr_perm_count[p_idx] == 1:
                    new_unique_perms.append((j, p_idx))
            
            # Update pos_critical for the affected windows
            # For each window that became unique, increment criticality for its positions
            for j, p_idx in new_unique_perms:
                for k in range(j, j + n):
                    if k < curr_L:
                        curr_pos_critical[k] += 1
            
            # Also, we need to decrement criticality for the deleted position?
            # No, pos_critical is a property of the *current* state.
            # After deletion, the state changes.
            # We are updating pos_critical to reflect the *new* state.
            # But we haven't deleted yet.
            # The "new unique perms" are unique in the *new* state.
            # So we add to their criticality.
            # What about perms that were unique before?
            # If a perm was unique (count 1) and covered del_idx, then pos_critical[del_idx] would have been >= 1.
            # So we didn't pick del_idx.
            # So no unique perms cover del_idx.
            # So no pos_critical values decrease due to this deletion?
            # Wait.
            # pos_critical[k] is the number of unique perms covering k.
            # If a perm p is unique and covers k, it contributes 1.
            # If we delete a char, and p is still present (count > 0), is it still unique?
            # If count[p] was 2 and becomes 1, it becomes unique. So we ADD to criticality.
            # If count[p] was 3 and becomes 2, no change.
            # If count[p] was 1... it can't be, because del_idx is not critical.
            
            # So we only ADD to criticality for the newly unique perms.
            
            # 2. Pop element
            curr_arr.pop(del_idx)
            curr_L -= 1
            
            # 3. Update perm_idx
            # Shift indices >= del_idx left by 1
            # new_perm_idx[i] = old_perm_idx[i+1] for i >= del_idx
            # But we only care about i in [0, curr_L - n]
            # The windows starting at i in new array correspond to i+1 in old array for i >= del_idx
            # For i < del_idx, the windows are unchanged, EXCEPT those that contained del_idx.
            # The windows starting at j in [del_idx - n + 1, del_idx - 1] in the new array
            # are formed by old_arr[j:del_idx] + old_arr[del_idx+1:j+n+1]
            # These need to be recalculated.
            
            # First, shift the tail
            # We can do this by slicing or loop
            # Note: curr_perm_idx has length L (original). We are reusing it.
            # We only use indices 0..curr_L-n.
            
            # Shift
            for i in range(curr_L - n, del_idx - 1, -1):
                curr_perm_idx[i] = curr_perm_idx[i+1]
                
            # Recalculate the affected head
            # Windows starting at j in [max(0, del_idx - n + 1), del_idx - 1]
            # Note: del_idx in new array is the position of the element that was at del_idx+1
            # The window starting at j in new array ends at j+n-1.
            # If j+n-1 < del_idx, it's unaffected.
            # If j < del_idx <= j+n-1, it's affected.
            # So j in [del_idx - n + 1, del_idx - 1].
            
            for j in range(max(0, del_idx - n + 1), del_idx):
                curr_perm_idx[j] = get_perm_idx(curr_arr, j)
                
            # 4. Update perm_count
            # We already updated perm_count for the windows that *existed* in the old array and contained del_idx.
            # But the windows that were *shifted* or *recalculated* might have different perm_ids.
            # Actually, the windows starting at j >= del_idx in new array are the same as j+1 in old array.
            # So their perm_id is the same. So perm_count for those perms is already correct?
            # Wait.
            # The set of windows in the new array is:
            # 1. Windows starting at j < del_idx - n + 1: Unchanged.
            # 2. Windows starting at j in [del_idx - n + 1, del_idx - 1]: Recalculated.
            # 3. Windows starting at j >= del_idx: Same as old windows at j+1.
            
            # The old windows that contained del_idx were j in [del_idx - n + 1, del_idx].
            # We decremented their counts.
            # The new windows in category 3 are a subset of the old windows (specifically, old windows j+1).
            # The new windows in category 2 are *new* windows (or modified).
            
            # This incremental count update is tricky.
            # It is safer to recalculate perm_count from scratch for the affected region?
            # Or just recalculate perm_count for all perms? No, too slow.
            
            # Alternative:
            # Since we only delete a few characters, and L is ~5900,
            # maybe we can just recalculate perm_count and pos_critical from scratch?
            # O(L * n) = 5900 * 7 = 41300 ops.
            # If we do this for each deletion, and we do ~100 deletions, that's 4 million ops.
            # In Python, 4 million ops is ~0.1s.
            # 1000 restarts * 0.1s = 100s. Too slow.
            
            # We need the incremental update to be correct.
            
            # Let's trace the counts.
            # Old state:
            # Windows W_old.
            # New state:
            # Windows W_new.
            
            # W_new consists of:
            # - W_old[0 : del_idx - n + 1] (unchanged)
            # - New windows N (recalculated)
            # - W_old[del_idx + 1 : L - n + 1] (shifted)
            
            # The windows W_old[del_idx - n + 1 : del_idx + 1] are GONE.
            # We decremented their counts.
            
            # The windows N are NEW. We need to ADD their counts.
            
            # So:
            # 1. Decrement counts for W_old[del_idx - n + 1 : del_idx + 1]. (Done)
            # 2. Add counts for N (the recalculated windows).
            
            # So we need to add the counts for the recalculated windows.
            
            for j in range(max(0, del_idx - n + 1), del_idx):
                p_idx = curr_perm_idx[j]
                if p_idx != -1:
                    curr_perm_count[p_idx] += 1
                    if curr_perm_count[p_idx] == 1:
                        # It became unique (or was unique and stayed unique? No, it was new)
                        # Wait, if it was in the old array, it might have been counted.
                        # But we decremented the old ones.
                        # If the new window has the same perm as an old window that we decremented,
                        # the net change is 0?
                        # No, the old window is gone. The new window is different content (usually).
                        # So we just add.
                        
                        # If count becomes 1, it is unique.
                        # We need to update pos_critical.
                        # But we already updated pos_critical for the "newly unique" from the decrement step?
                        # No, the decrement step identified perms that became unique *because of the decrement*.
                        # These are perms that had count 2 and became 1.
                        # The addition step might create perms that become unique (count 0 -> 1? No, count was >= 0).
                        # If a perm had count 0, and we add 1, it becomes 1 (unique).
                        # If a perm had count 1, and we add 1, it becomes 2 (not unique).
                        
                        # This is getting complicated.
                        pass
            
            # Given the complexity and risk of bugs in incremental update,
            # and the fact that L is only 5913,
            # let's try a hybrid:
            # Recalculate perm_idx and perm_count for the whole array, but use a fast C-based method?
            # Or just optimize the Python loop.
            
            # Actually, let's just recalculate perm_count and pos_critical from scratch.
            # It's safer.
            
            # Recalculate perm_count
            # Reset
            for i in range(num_perms):
                curr_perm_count[i] = 0
            for i in range(curr_L - n + 1):
                if curr_perm_idx[i] != -1:
                    curr_perm_count[curr_perm_idx[i]] += 1
            
            # Recalculate pos_critical
            curr_pos_critical = [0] * curr_L
            for i in range(curr_L - n + 1):
                if curr_perm_idx[i] != -1 and curr_perm_count[curr_perm_idx[i]] == 1:
                    for k in range(n):
                        curr_pos_critical[i+k] += 1
            
            if curr_L < best_len:
                best_len = curr_L
                best_arr = curr_arr[:]
                
            if curr_L == target_len:
                break
        
        if best_len == target_len:
            break
            
        attempts += 1
        
    return ''.join(str(x) for x in best_arr)
