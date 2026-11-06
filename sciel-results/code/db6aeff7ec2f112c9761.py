import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    # Precompute conflicts: for each x, which sums it would create
    # Sums needed for a Sidon set: all a+b for a<=b in S must be distinct
    
    def init_sums(S):
        Sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                Sums.add(S[i] + S[j])
        return Sums
    
    def can_add(x, S, Sums):
        # Check if adding x creates duplicate sums
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True
    
    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()
    
    def remove_element(x, S, Sums):
        S.remove(x)
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)
    
    def get_conflicts(x, S, Sums):
        """Return set of elements in S that conflict with x."""
        conflicts = set()
        if 2 * x in Sums:
            # 2*x is a sum, find which pair creates it
            for i in range(len(S)):
                for j in range(i, len(S)):
                    if S[i] + S[j] == 2 * x:
                        conflicts.add(S[i])
                        conflicts.add(S[j])
                        break
        for y in S:
            if x + y in Sums:
                # Find which pair creates x+y
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        if S[i] + S[j] == x + y:
                            conflicts.add(S[i])
                            conflicts.add(S[j])
                            break
        return conflicts
    
    def greedy_fill(S, Sums, order='ascending'):
        """Fill S greedily."""
        S_set = set(S)
        if order == 'ascending':
            candidates = list(range(1, max_val + 1))
        elif order == 'descending':
            candidates = list(range(max_val, 0, -1))
        else:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if x in S_set:
                continue
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
        return S, Sums
    
    def local_search(S, Sums, max_iters=200, aggressive=False):
        """Local search with adds, swaps, and double swaps."""
        S_set = set(S)
        improved = False
        iters = 0
        stuck_count = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            # Try to add an element - check all candidates systematically
            added = False
            for x in all_vals:
                if x in S_set:
                    continue
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
            
            stuck_count += 1
            
            # If stuck, try swaps - try all removable elements
            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break
            
            swap_done = False
            
            # Try single swaps - try each removable element
            for r in removable:
                if time.time() >= deadline - 5:
                    break
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                # Try to find an element that can be added
                found = False
                for x in all_vals:
                    if x in S_set or x == r:
                        continue
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        swap_done = True
                        improved = True
                        stuck_count = 0
                        found = True
                        break
                
                if found:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)
            
            if swap_done:
                continue
            
            # Try double swaps if aggressive
            if aggressive and len(removable) >= 2:
                # Try pairs of removable elements
                pair_found = False
                for i in range(len(removable)):
                    if time.time() >= deadline - 5:
                        break
                    for j in range(i + 1, len(removable)):
                        r1, r2 = removable[i], removable[j]
                        
                        remove_element(r1, S, Sums)
                        S_set.discard(r1)
                        remove_element(r2, S, Sums)
                        S_set.discard(r2)
                        
                        # Try to add two elements
                        added_count = 0
                        added_elems = []
                        for x in all_vals:
                            if x in S_set or x == r1 or x == r2:
                                continue
                            if can_add(x, S, Sums):
                                add_element(x, S, Sums)
                                S_set.add(x)
                                added_elems.append(x)
                                added_count += 1
                                if added_count >= 2:
                                    break
                        
                        if added_count >= 2:
                            improved = True
                            stuck_count = 0
                            pair_found = True
                            break
                        else:
                            for x in added_elems:
                                remove_element(x, S, Sums)
                                S_set.discard(x)
                            add_element(r1, S, Sums)
                            S_set.add(r1)
                            add_element(r2, S, Sums)
                            S_set.add(r2)
                
                if pair_found:
                    continue
            
            if stuck_count > 30:
                break
        
        return S, Sums, improved
    
    def hill_climbing_polish(S, Sums):
        """Final polish: try all possible single moves to improve."""
        S_set = set(S)
        improved = True
        polish_iters = 0
        
        while improved and polish_iters < 50 and time.time() < deadline - 3:
            improved = False
            polish_iters += 1
            
            # Try all adds
            for x in all_vals:
                if x in S_set:
                    continue
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try all single swaps - only if we have time
            if time.time() >= deadline - 3:
                break
                
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                if time.time() >= deadline - 3:
                    break
                    
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                for x in all_vals:
                    if x in S_set or x == r:
                        continue
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        improved = True
                        break
                
                if improved:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)
        
        return S, Sums
    
    best_size = 0
    best_S = []
    
    # Strategy: Multiple greedy + local search restarts with different orders
    restart_count = 0
    
    while time.time() < deadline - 15:
        restart_count += 1
        
        # Choose strategy based on restart number
        if restart_count % 4 == 1:
            order = 'ascending'
        elif restart_count % 4 == 2:
            order = 'descending'
        elif restart_count % 4 == 3:
            order = 'random'
        else:
            # Try a mixed approach: start ascending, then random
            order = 'ascending'
        
        S = list(prescribed)
        S.sort()
        Sums = init_sums(S)
        S, Sums = greedy_fill(S, Sums, order=order)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
        
        # Local search with more iterations
        aggressive = restart_count % 2 == 0
        S, Sums, improved = local_search(S, Sums, max_iters=200, aggressive=aggressive)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
        
        # If we have time, do another local search pass on the best solution
        if time.time() < deadline - 20 and best_S:
            S2 = best_S.copy()
            Sums2 = init_sums(S2)
            S2, Sums2, _ = local_search(S2, Sums2, max_iters=100, aggressive=True)
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = S2.copy()
    
    # Final polish on best solution
    if best_S and time.time() < deadline - 5:
        S = best_S.copy()
        Sums = init_sums(S)
        S, Sums = hill_climbing_polish(S, Sums)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    return best_S
