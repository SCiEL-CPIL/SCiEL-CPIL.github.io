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
        
        def dfs(elements, used_diffs, covered_mask, last_elem):
            if time.time() > deadline:
                return False
            
            if covered_mask == TARGET_MASK:
                result[0] = elements[:]
                return True
            
            remaining_mask = TARGET_MASK & ~covered_mask
            remaining_count = bin(remaining_mask).count('1')
            
            # Pruning: need at least ceil(remaining_count / len(elements)) more elements
            # and they must fit in (last_elem, D]
            min_elements_needed = (remaining_count + len(elements) - 1) // len(elements)
            if D - last_elem < min_elements_needed:
                return False
            
            # Generate candidates in order of coverage
            candidates = []
            for x in range(last_elem + 1, D + 1):
                valid = True
                new_covered = 0
                for y in elements:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    if d <= 30:
                        bit = 1 << (d - 1)
                        if bit & remaining_mask:
                            new_covered += 1
                if valid:
                    candidates.append((new_covered, x))
            
            # Sort by coverage descending, then x ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for covered, x in candidates:
                if covered == 0 and len(candidates) > 5:
                    continue  # Skip low-value candidates if we have better options
                
                new_used_diffs = set(used_diffs)
                new_covered_mask = covered_mask
                for y in elements:
                    d = x - y
                    new_used_diffs.add(d)
                    if d <= 30:
                        new_covered_mask |= (1 << (d - 1))
                
                new_elements = elements + [x]
                if dfs(new_elements, new_used_diffs, new_covered_mask, x):
                    return True
            
            return False
        
        dfs([0], set(), 0, 0)
        return result[0]
    
    # Try increasing D from lower bound
    # Lower bound: need at least 9 elements (0 + 8 others), each adding ~8 diffs max
    # 30 diffs / 8 per element ~ 4 elements min, but Sidon constraints make it harder
    # Empirically, D >= ~50-60 should work
    for D in range(50, 200):
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
