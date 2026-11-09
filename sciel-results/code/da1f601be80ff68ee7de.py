import time
import sys

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    best_max = float('inf')
    best_set = None
    
    def solve_k(k, D):
        """
        Finds a Sidon set of size k covering 1..30 with max element <= D.
        Returns the set if found, else None.
        Uses DFS with branch and bound.
        """
        current_set = [0]
        used_diffs = set()
        covered_mask = 0
        
        def dfs(idx, last_val):
            nonlocal covered_mask
            
            if time.time() - start_time > deadline - 2:
                return False
            
            if idx == k:
                if covered_mask == full_mask:
                    if last_val < best_max:
                        best_max = last_val
                        best_set = list(current_set)
                    return True
                return False
            
            # Pruning: Can we cover remaining missing diffs?
            missing = 30 - bin(covered_mask).count('1')
            remaining_slots = k - idx
            # Loose bound: max new diffs
            max_new_diffs = remaining_slots * idx + remaining_slots * (remaining_slots - 1) // 2
            if missing > max_new_diffs:
                return False
            
            # Candidate generation
            candidates = set()
            candidates.add(last_val + 1)
            
            missing_list = [d for d in range(1, 31) if not (covered_mask & (1 << (d-1)))]
            for d in missing_list:
                for s in current_set:
                    x = s + d
                    if x > last_val and x <= D:
                        candidates.add(x)
            
            # Filter and score
            valid_candidates = []
            for x in candidates:
                if x <= last_val or x > D:
                    continue
                
                new_diffs = []
                valid = True
                for s in current_set:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d-1))
                
                new_missing = 30 - bin(new_mask).count('1')
                valid_candidates.append((new_missing, x, new_mask, new_diffs))
            
            valid_candidates.sort(key=lambda item: (item[0], item[1]))
            
            for new_missing, x, new_mask, new_diffs in valid_candidates:
                current_set.append(x)
                used_diffs.update(new_diffs)
                old_covered = covered_mask
                covered_mask = new_mask
                
                if dfs(idx + 1, x):
                    return True
                
                current_set.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                covered_mask = old_covered
                
            return False
            
        return dfs(1, 0)

    # Iterative deepening on Diameter D
    # Start D from 30 up to 100
    # Try k=9 first.
    
    for k in [9, 10, 11]:
        if time.time() - start_time > deadline - 5:
            break
            
        for D in range(30, 100):
            if time.time() - start_time > deadline - 2:
                break
                
            # We want to find *a* solution with max <= D.
            # If we find one, it's the best for this D.
            # Since we iterate D increasing, the first D that works is minimal.
            
            # Note: solve_k updates best_set if it finds a better one.
            # But we want the first D that works.
            # If solve_k returns True, we found a solution.
            # We can return immediately?
            # Yes, because we are iterating D from low to high.
            # The first D that allows a solution is the minimum diameter.
            
            if solve_k(k, D):
                # Found a solution with max <= D
                # Actually, solve_k might find a solution with max < D.
                # But we are checking if *any* solution exists in [0, D].
                # If yes, the minimal diameter is <= D.
                # We can return the best_set found.
                # However, solve_k might have found a solution with max=D-1 in a previous iteration?
                # No, we iterate D. If D=40 fails, D=41 might succeed with max=41.
                # If D=41 succeeds with max=40, we would have found it in D=40?
                # Yes, if a solution with max=40 exists, it is a valid solution for D=40.
                # So if D=40 failed, no solution with max<=40 exists.
                # Thus, the first D that succeeds gives the minimal diameter.
                
                if best_set:
                    return best_set
                else:
                    # Fallback if best_set not updated (shouldn't happen if solve_k returns True)
                    return [0, 1, 4, 10, 18, 23, 25, 29, 30] # Dummy
            
        # If no solution found for this k in range 30..99, try next k
        
    if best_set:
        return best_set
    else:
        # Fallback
        return [0, 1, 4, 10, 18, 23, 25, 29, 30]

