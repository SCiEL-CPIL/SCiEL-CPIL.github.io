import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Helper to rebuild diffs and sums for a set
    def rebuild_diffs_sums(S):
        diffs = set()
        sums = set()
        for i in range(len(S)):
            a = S[i]
            sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                d = abs(a - b)
                diffs.add(d)
                sums.add(a + b)
        return diffs, sums
    
    # Fast check for adding x to set S
    def can_add_fast(x, S, diffs_S, sums_S, diffs_other):
        if x in S:
            return False
            
        # Check sums (Sidon property)
        if 2 * x in sums_S:
            return False
        for a in S:
            if (x + a) in sums_S:
                return False
                
        # Check diffs against other set
        for a in S:
            d = abs(x - a)
            if d in diffs_other:
                return False
                
        return True
    
    # Add x to set S, updating diffs_S and sums_S
    def add_to_set_fast(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)
    
    def greedy_construct(seed_val):
        random.seed(seed_val)
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            valid_A = can_add_fast(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add_fast(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set_fast(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set_fast(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set_fast(x, A, diffs_A, sums_A)
                    else:
                        add_to_set_fast(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set_fast(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set_fast(x, B, diffs_B, sums_B)
        
        return A, B, diffs_A, diffs_B, sums_A, sums_B
    
    def local_search(A, B, diffs_A, diffs_B, sums_A, sums_B, time_limit):
        """
        Local search with 2-for-1 swaps to break local optima.
        Returns improved (A, B, diffs_A, diffs_B, sums_A, sums_B).
        """
        # Work on copies to avoid modifying original
        A = list(A)
        B = list(B)
        diffs_A = set(diffs_A)
        diffs_B = set(diffs_B)
        sums_A = set(sums_A)
        sums_B = set(sums_B)
        
        unplaced = [x for x in range(1, 257) if x not in A and x not in B]
        
        # Phase 1: Add unplaced elements
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            random.shuffle(unplaced)
            for x in list(unplaced):
                valid_A = can_add_fast(x, A, diffs_A, sums_A, diffs_B)
                valid_B = can_add_fast(x, B, diffs_B, sums_B, diffs_A)
                
                if valid_A:
                    add_to_set_fast(x, A, diffs_A, sums_A)
                    unplaced.remove(x)
                    changed = True
                elif valid_B:
                    add_to_set_fast(x, B, diffs_B, sums_B, diffs_A)
                    unplaced.remove(x)
                    changed = True
        
        # Phase 2: 2-for-1 swaps
        # Try to remove one element from a set and add two from unplaced
        for _ in range(50):
            if time.time() > time_limit:
                break
            if len(unplaced) < 2:
                break
            
            # Pick which set to modify
            modify_A = random.random() < 0.5
            if modify_A:
                S = A
                other_S = B
                diffs_S = diffs_A
                sums_S = sums_A
                diffs_other = diffs_B
            else:
                S = B
                other_S = A
                diffs_S = diffs_B
                sums_S = sums_B
                diffs_other = diffs_A
            
            if len(S) < 1:
                continue
            
            # Pick random element to remove
            idx = random.randint(0, len(S) - 1)
            x_remove = S[idx]
            
            # Save state
            S_backup = list(S)
            diffs_S_backup = set(diffs_S)
            sums_S_backup = set(sums_S)
            
            # Remove
            S.pop(idx)
            new_diffs_S, new_sums_S = rebuild_diffs_sums(S)
            
            # Try to find 2 elements from unplaced to add
            found = False
            if len(unplaced) >= 2:
                # Pick random pairs from unplaced
                i1, i2 = random.sample(range(len(unplaced)), 2)
                y1, y2 = unplaced[i1], unplaced[i2]
                
                # Check if y1 can be added
                if can_add_fast(y1, S, new_diffs_S, new_sums_S, diffs_other):
                    # Add y1 to temp
                    temp_S = list(S)
                    temp_diffs_S = set(new_diffs_S)
                    temp_sums_S = set(new_sums_S)
                    add_to_set_fast(y1, temp_S, temp_diffs_S, temp_sums_S)
                    
                    # Check if y2 can be added
                    if can_add_fast(y2, temp_S, temp_diffs_S, temp_sums_S, diffs_other):
                        # Success - commit the change
                        S.clear()
                        S.extend(temp_S)
                        diffs_S.clear()
                        diffs_S.update(temp_diffs_S)
                        sums_S.clear()
                        sums_S.update(temp_sums_S)
                        unplaced.remove(y1)
                        unplaced.remove(y2)
                        found = True
            
            if not found:
                # Revert
                S.clear()
                S.extend(S_backup)
                diffs_S.clear()
                diffs_S.update(diffs_S_backup)
                sums_S.clear()
                sums_S.update(sums_S_backup)
        
        # Phase 3: Final greedy add after swaps
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            random.shuffle(unplaced)
            for x in list(unplaced):
                valid_A = can_add_fast(x, A, diffs_A, sums_A, diffs_B)
                valid_B = can_add_fast(x, B, diffs_B, sums_B, diffs_A)
                
                if valid_A:
                    add_to_set_fast(x, A, diffs_A, sums_A)
                    unplaced.remove(x)
                    changed = True
                elif valid_B:
                    add_to_set_fast(x, B, diffs_B, sums_B, diffs_A)
                    unplaced.remove(x)
                    changed = True
        
        return A, B, diffs_A, diffs_B, sums_A, sums_B
    
    # Main search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Greedy construction
        A, B, diffs_A, diffs_B, sums_A, sums_B = greedy_construct(attempts * 12345 + 67890)
        
        # Local search with 2-for-1 swaps
        ls_time_limit = min(time.time() + 2.0, deadline - 1.0)
        A, B, diffs_A, diffs_B, sums_A, sums_B = local_search(
            A, B, diffs_A, diffs_B, sums_A, sums_B, ls_time_limit
        )
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
            
            # If we've reached a good score, we can be more aggressive with time
            if best_score >= 40:
                break
    
    return [best_A, best_B]
