# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    """
    m = 205
    k = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s buffer for safety
    
    # Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute conflict matrix: conflict[i][j] is True if (i-j) or (j-i) mod m is in forbidden
    # Actually, we just need to check if difference is in forbidden.
    # conflict[i][j] = True if i and j cannot both be in the set.
    # i and j conflict if (i - j) % m in forbidden OR (j - i) % m in forbidden.
    # Note: if d is in forbidden, -d mod m might be different. But forbidden is symmetric?
    # If x^2 = d, then (-x)^2 = d. So d is a square.
    # Is -d a square? (-1) * d. Depends on -1 being a square.
    # 205 = 5 * 41.
    # Legendre symbol (-1/p): -1 is square mod p if p=1 mod 4.
    # 5 = 1 mod 4, so -1 is square mod 5.
    # 41 = 1 mod 4, so -1 is square mod 41.
    # By CRT, -1 is square mod 205.
    # So if d is a square, -d is a square.
    # Thus forbidden is symmetric: d in forbidden iff (-d) mod m in forbidden.
    # So we just need to check if (i - j) % m in forbidden.
    
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True
    
    # Helper to check if a set of indices is valid
    def is_valid(indices):
        indices = list(indices)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if conflict[indices[a]][indices[b]]:
                    return False
        return True
    
    # Helper to get size
    def size(indices):
        return len(indices)
    
    # Greedy initialization
    def greedy_init():
        selected = []
        remaining = list(range(m))
        random.shuffle(remaining)
        for candidate in remaining:
            ok = True
            for s in selected:
                if conflict[candidate][s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    # Local search: try to swap elements to improve size
    # We represent the set as a boolean array or a list.
    # Let's use a set for O(1) lookup, but we need to check conflicts.
    # Better: maintain a list of selected elements.
    # Move: remove one element, add another.
    # Or: try to add an element not in set, if conflict, try to remove conflicting ones.
    
    def local_search(initial_set):
        current_set = set(initial_set)
        current_size = len(current_set)
        
        # List of all elements not in current set
        not_in_set = [i for i in range(m) if i not in current_set]
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try all possible additions
            random.shuffle(not_in_set)
            for new_elem in not_in_set:
                if time.time() > deadline - 10:
                    break
                # Check if new_elem conflicts with any in current_set
                conflicts_with = [s for s in current_set if conflict[new_elem][s]]
                if len(conflicts_with) == 0:
                    # Can add directly
                    current_set.add(new_elem)
                    current_size += 1
                    not_in_set.remove(new_elem)
                    improved = True
                    # Restart shuffle or continue? Continue is fine, but we modified not_in_set.
                    # Let's just break and restart the loop for simplicity/safety
                    break
                else:
                    # Try to remove one of the conflicts and add new_elem
                    # This is a swap.
                    # We need to ensure that after removing s, new_elem doesn't conflict with others.
                    # Since new_elem only conflicted with conflicts_with, removing any s in conflicts_with
                    # makes it compatible with the rest?
                    # Wait, new_elem might conflict with multiple elements in conflicts_with.
                    # If we remove s1, does new_elem conflict with s2 (where s2 is also in conflicts_with)?
                    # Yes, if s2 is in conflicts_with, it conflicts.
                    # So we can only add new_elem if we remove ALL elements it conflicts with.
                    # That's a "delete many, add one" move.
                    # Or we can try 1-for-1 swaps if new_elem conflicts with exactly 1 element.
                    
                    if len(conflicts_with) == 1:
                        s_to_remove = conflicts_with[0]
                        # Perform swap
                        current_set.remove(s_to_remove)
                        current_set.add(new_elem)
                        not_in_set.remove(new_elem)
                        not_in_set.append(s_to_remove)
                        improved = True
                        break
                    elif len(conflicts_with) == 2:
                        # Try removing both? That decreases size by 1. Not good for maximization.
                        # Unless we can add 2 elements?
                        # For now, stick to 0-conflict adds and 1-conflict swaps.
                        pass
                    # Higher order moves are complex.
            
            # Also try to add elements if possible after shuffling
            # The loop above handles adds.
            
            # Try 2-for-2 swaps?
            # Too expensive for now.
            
            # If no improvement in a full pass, stop.
            
            # Simple restarts if stuck?
            # Let's add a simple random restart if stuck for a while.
            # But we need to track "stuck".
            # For now, just break if no improvement.
            
            # To prevent infinite loop if no improvement, we rely on 'improved' flag.
            # But if we break out of 'for new_elem' without improvement, we exit while.
            # We should maybe do multiple passes.
            
            # Let's restructure:
            # Pass 1: try all adds/swaps.
            # If none, try random restart?
            
            # Actually, the logic above breaks out of 'for' if improved.
            # If not improved, it finishes the loop and exits 'while'.
            # This is a single pass local search.
            
            # Let's do multiple passes.
            if not improved:
                break
                
        return list(current_set)
    
    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    num_restarts = 10000
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        init_set = greedy_init()
        size_init = len(init_set)
        
        # Local search
        ls_set = local_search(init_set)
        size_ls = len(ls_set)
        
        if size_ls > best_size:
            best_size = size_ls
            best_set = ls_set
            # Optional: if we hit a known good size, we might continue to find better.
            # Max possible size is unknown, but 12 is a record mentioned.
            # If we find 13, great.
            
        # Also try improving the greedy result directly with more aggressive local search?
        # The local_search above is quite basic.
        
        # Alternative: Simulated Annealing?
        # Let's stick to greedy + local search for now, but maybe enhance local search.
        
        # Enhanced local search:
        # 1. Add non-conflicting elements.
        # 2. Swap 1 element to free up space for another?
        # 3. Try to add element, if conflicts, try to remove a conflicting element and see if size increases?
        #    Removing 1 and adding 1 keeps size same. We need to add 1 and remove 0 (size+1) or add 2 and remove 1 (size+1).
        
        # Let's implement a more thorough local search.
        
        def deep_local_search(initial_set):
            current_set = set(initial_set)
            current_size = len(current_set)
            
            # Precompute conflict lists for speed
            # conflict_list[i] = list of j such that conflict[i][j] is True
            conflict_list = [[] for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    if conflict[i][j]:
                        conflict_list[i].append(j)
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                not_in_set = [i for i in range(m) if i not in current_set]
                random.shuffle(not_in_set)
                
                for new_elem in not_in_set:
                    if time.time() > deadline - 10:
                        return list(current_set)
                    
                    # Find conflicts in current_set
                    conflicts_in_set = [s for s in current_set if conflict[new_elem][s]]
                    
                    if len(conflicts_in_set) == 0:
                        current_set.add(new_elem)
                        current_size += 1
                        improved = True
                        break
                    elif len(conflicts_in_set) == 1:
                        # Swap
                        s_to_remove = conflicts_in_set[0]
                        # Check if removing s_to_remove allows new_elem
                        # Yes, by definition.
                        # But does it break anything else? No, we are just swapping.
                        current_set.remove(s_to_remove)
                        current_set.add(new_elem)
                        # Size remains same.
                        # This move doesn't increase size, but might enable future increases.
                        # So we should accept it if it leads to improvement later.
                        # For simplicity, let's only accept size-increasing moves in the main loop.
                        # But we can do a "swap" pass to rearrange.
                        # Let's skip pure swaps for now to keep it simple and focused on size increase.
                        pass
                    elif len(conflicts_in_set) == 2:
                        # Try to remove 2 and add 1? Size -1. No.
                        # Try to remove 1 and add 1? Still conflicts with the other.
                        # So we can't add new_elem unless we remove all conflicts.
                        # If we remove all conflicts, size changes by 1 - len(conflicts).
                        # So we need len(conflicts) == 0 to increase size.
                        # Or len(conflicts) == 1 and we swap (size same).
                        # Or len(conflicts) == 2 and we add 2 elements?
                        
                        # Let's try a 2-for-1 move:
                        # Find another element new_elem2 not in set, not conflicting with new_elem,
                        # such that removing the 2 conflicts allows adding both.
                        # This is complex.
                        
                        pass
                    
                    # If we can't add, try to find a pair of elements to remove that allows adding 2?
                    # Too complex.
                
                # If no improvement in this pass, try some random swaps to escape local minima?
                # Or just break.
                if not improved:
                    # Try a few random swaps
                    for _ in range(10):
                        if time.time() > deadline - 10:
                            break
                        if len(current_set) < 2:
                            break
                        a, b = random.sample(list(current_set), 2)
                        # Try to replace a with some c not in set
                        # Or just swap a and b? No, that does nothing.
                        # Replace a with c.
                        # Check if c conflicts with current_set - {a}
                        # And c is not in set.
                        # And c doesn't conflict with b (which is in set).
                        
                        # Let's just pick a random c not in set.
                        c = random.choice([i for i in range(m) if i not in current_set])
                        
                        # Check conflicts
                        ok = True
                        for s in current_set:
                            if s != a and conflict[c][s]:
                                ok = False
                                break
                        if ok:
                            current_set.remove(a)
                            current_set.add(c)
                            # Size same.
                            # Check if this enables an add?
                            # We'll let the next loop iteration handle adds.
                            improved = True
                            break
                            
            return list(current_set)
        
        # Use the deeper local search
        ls_set2 = deep_local_search(init_set)
        size_ls2 = len(ls_set2)
        
        if size_ls2 > best_size:
            best_size = size_ls2
            best_set = ls_set2
            
        # Also try starting from the best set found so far and doing more local search
        if i % 10 == 0 and best_set:
            ls_set3 = deep_local_search(best_set)
            size_ls3 = len(ls_set3)
            if size_ls3 > best_size:
                best_size = size_ls3
                best_set = ls_set3

    # Final verification
    if best_set:
        # Ensure no duplicates
        best_set = list(set(best_set))
        # Ensure valid
        if not is_valid(best_set):
             # Fallback to a simple valid set
             fallback = [0]
             return np.array(fallback, dtype=np.int64)
        
        return np.array(sorted(best_set), dtype=np.int64)
    else:
        return np.array([0], dtype=np.int64)


# EVOLVE-BLOCK-END
