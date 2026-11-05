import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 192
    
    # Initial set: Numbers in upper half - they have fewer multiples in range
    # Try multiple initial strategies and pick the best
    initial_candidates = [
        set(range(N // 2 + 1, N + 1)),  # (96, 192]
        set(range(N // 3 + 1, N + 1)),  # (64, 192]
        set(range(N // 4 + 1, N + 1)),  # (48, 192]
    ]
    
    def find_violations_fast(S):
        """Find all violations: (a, u, v, w) where a divides u+v+w, all distinct"""
        violations = []
        S_list = list(S)
        n = len(S_list)
        
        for idx_a, a in enumerate(S_list):
            # T is S without a
            T = [x for x in S_list if x != a]
            if len(T) < 3:
                continue
            
            # Map residues to elements in T
            res_map = {}
            for val in T:
                r = val % a
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(val)
            
            # Iterate pairs in T
            for i in range(len(T)):
                u = T[i]
                for j in range(i+1, len(T)):
                    v = T[j]
                    s2 = u + v
                    target_r = (-s2) % a
                    if target_r in res_map:
                        candidates = res_map[target_r]
                        for w in candidates:
                            if w != u and w != v:
                                violations.append((a, u, v, w))
                                break
        return violations
    
    def is_valid(S):
        return len(find_violations_fast(S)) == 0
    
    # Try each initial strategy, repair, and keep the best
    best_S = set()
    best_size = 0
    
    for initial in initial_candidates:
        if time.time() > deadline:
            break
            
        S = set(initial)
        
        # Repair Phase
        max_repair_steps = 200
        for step in range(max_repair_steps):
            if time.time() > deadline:
                break
                
            violations = find_violations_fast(S)
            if not violations:
                break
            
            # Count frequency of each element in violations
            freq = {}
            for v in violations:
                for elem in v:
                    freq[elem] = freq.get(elem, 0) + 1
            
            # Pick element with max frequency
            # If tie, pick smallest value (more likely to cause future issues)
            worst_elem = min(freq.keys(), key=lambda x: (-freq[x], x))
            
            S.remove(worst_elem)
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    S = best_S
    removed = set(range(1, N+1)) - S
    
    # Backfill Phase with multiple passes
    removed_list = sorted(list(removed), reverse=True)
    
    def try_add(S, x):
        """Check if S U {x} is valid. Returns True if valid."""
        if x in S:
            return False
        
        # Check 1: x does not divide sum of any 3 in S
        if len(S) >= 3:
            T = list(S)
            res_map = {}
            for val in T:
                r = val % x
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(val)
            
            for i in range(len(T)):
                u = T[i]
                for j in range(i+1, len(T)):
                    v = T[j]
                    s2 = u + v
                    target_r = (-s2) % x
                    if target_r in res_map:
                        for w in res_map[target_r]:
                            if w != u and w != v:
                                return False
        
        # Check 2: No a in S divides sum of any 3 involving x
        # Sums are x + u + v for u, v in S distinct
        if len(S) >= 2:
            for a in S:
                target_r = (-x) % a
                T = list(S)
                found = False
                for i in range(len(T)):
                    u = T[i]
                    for j in range(i+1, len(T)):
                        v = T[j]
                        if (x + u + v) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True
    
    backfill_passes = 10
    for p in range(backfill_passes):
        if time.time() > deadline:
            break
        progress = False
        for x in removed_list:
            if x in S:
                continue
            if try_add(S, x):
                S.add(x)
                removed.remove(x)
                progress = True
        if not progress:
            break
    
    # Final cleanup
    while time.time() < deadline:
        violations = find_violations_fast(S)
        if not violations:
            break
        freq = {}
        for v in violations:
            for elem in v:
                freq[elem] = freq.get(elem, 0) + 1
        worst_elem = min(freq.keys(), key=lambda x: (-freq[x], x))
        S.remove(worst_elem)
        removed.add(worst_elem)
    
    return list(S)
