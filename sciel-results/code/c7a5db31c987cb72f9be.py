import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    sys.setrecursionlimit(10000)
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 represent diffs 1..30
    
    def is_valid(a):
        if 0 not in a:
            return False
        if len(a) > 32:
            return False
        if max(a) > 10**6:
            return False
        diffs = set()
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    # Greedy fallback
    def greedy_construct():
        a = [0]
        used_diffs = set()
        missing = set(range(1, 31))
        for _ in range(32):
            if not missing:
                break
            best_x = None
            best_score = -1
            for x in range(max(a) + 1, 300):
                new_diffs = [x - y for y in a]
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                covered = sum(1 for d in new_diffs if d in missing)
                if covered > best_score:
                    best_score = covered
                    best_x = x
            if best_x is None:
                break
            a.append(best_x)
            for y in a[:-1]:
                d = best_x - y
                used_diffs.add(d)
                if d in missing:
                    missing.remove(d)
        return a if not missing else None
    
    init_sol = greedy_construct()
    best_sol = init_sol if init_sol else None
    best_max_val = max(init_sol) if init_sol else float('inf')
    
    def solve_for_D(D):
        """Find a Sidon set in [0,D] containing 0, covering all diffs 1..30."""
        result = [None]
        found = [False]
        
        def dfs(elements, used_diffs, covered_mask, last_elem):
            if found[0]:
                return True
            if time.time() > deadline:
                return False
            
            if covered_mask == TARGET_MASK:
                result[0] = elements[:]
                found[0] = True
                return True
            
            remaining_mask = TARGET_MASK & ~covered_mask
            remaining_count = bin(remaining_mask).count('1')
            
            # Pruning 1: Need at least ceil(remaining_count / len(elements)) more elements
            min_elements_needed = (remaining_count + len(elements) - 1) // len(elements)
            if D - last_elem < min_elements_needed:
                return False
            
            # Pruning 2: Check if any remaining diff is impossible to cover
            # For each remaining diff d, we need elements a,b in [0,D] with a-b=d
            # and both a,b not causing conflicts. At minimum, d <= D.
            # All remaining diffs are <= 30, and D >= 30, so this is fine.
            
            # Pruning 3: Gap analysis - if we have gaps > 30 between consecutive elements,
            # we can't cover diffs that span those gaps
            # Actually, diffs are between any pair, not just consecutive
            
            # Generate candidates: try x from last_elem+1 to D
            # Score by how many new diffs they cover
            candidates = []
            for x in range(last_elem + 1, D + 1):
                valid = True
                new_covered = 0
                new_used = set()
                for y in elements:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_used.add(d)
                    if d <= 30:
                        bit = 1 << (d - 1)
                        if bit & remaining_mask:
                            new_covered += 1
                if valid:
                    candidates.append((new_covered, x, new_used))
            
            # Sort by coverage descending, then x ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to improve performance
            max_candidates = min(len(candidates), 50)
            
            for i in range(max_candidates):
                covered, x, new_used = candidates[i]
                
                # Skip if this candidate covers nothing and we have better options
                if covered == 0 and i > 0:
                    continue
                
                new_used_diffs = used_diffs | new_used
                new_covered_mask = covered_mask
                for d in new_used:
                    if d <= 30:
                        new_covered_mask |= (1 << (d - 1))
                
                new_elements = elements + [x]
                if dfs(new_elements, new_used_diffs, new_covered_mask, x):
                    return True
            
            return False
        
        dfs([0], set(), 0, 0)
        return result[0]
    
    # Try increasing D from lower bound
    # For a Sidon set covering diffs 1..30, we need enough elements
    # Each new element adds at most len(elements) new diffs
    # With k elements total, we have k*(k-1)/2 diffs
    # Need at least 30 diffs, so k*(k-1)/2 >= 30 => k >= 9
    # But Sidon constraints make it harder
    # Empirically, D around 50-70 should work
    for D in range(45, 200):
        if time.time() > deadline:
            break
        
        sol = solve_for_D(D)
        if sol is not None:
            if is_valid(sol):
                if max(sol) < best_max_val:
                    best_max_val = max(sol)
                    best_sol = sol
                return sol
    
    # Return best found
    if best_sol is not None:
        return best_sol
    
    # Final fallback
    return greedy_construct()
