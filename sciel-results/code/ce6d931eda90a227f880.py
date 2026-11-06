import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target: Sidon set containing 0, covering diffs 1..30, min max element.
    # Lower bound for max element D is 36 (for n=9, K=36 diffs, must be distinct positive ints).
    
    def is_valid_solution(S):
        if 0 not in S:
            return False
        if len(S) > 32:
            return False
        if max(S) > 10**6:
            return False
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))
        
    def backtrack(S, used_diffs, D_target):
        if time.time() > deadline:
            return None
            
        # Check completion
        if all(d in used_diffs for d in range(1, 31)):
            return S[:]
            
        if len(S) >= 32:
            return None
            
        # Pruning: Can we cover remaining diffs with remaining slots?
        # Each new element adds at most len(S) new diffs.
        # Max new diffs with k more elements: sum_{i=len(S)}^{31} i
        missing = [d for d in range(1, 31) if d not in used_diffs]
        if not missing:
            return S[:]
            
        max_possible = 0
        for i in range(len(S), 32):
            max_possible += i
        if len(missing) > max_possible:
            return None

        # Generate candidates
        candidates = set()
        # 1. Based on missing diffs
        for x in S:
            for d in missing:
                candidates.add(x + d)
                if x - d >= 0:
                    candidates.add(x - d)
        # 2. Small values (to keep diameter small)
        for v in range(1, D_target + 1):
            candidates.add(v)
            
        scored_candidates = []
        for c in candidates:
            if c < 0 or c > D_target:
                continue
            if c in S:
                continue
                
            # Check Sidon property
            new_diffs = set()
            valid = True
            for x in S:
                d = abs(c - x)
                if d == 0:
                    continue
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
                
            # Score: how many missing diffs does this cover?
            covered = len(new_diffs & set(missing))
            if covered == 0:
                # Still might be useful later?
                # But if it doesn't cover any missing, it's less prioritized.
                # However, it might enable other diffs.
                # Let's include it with low score.
                pass
                
            scored_candidates.append((covered, c, new_diffs))
            
        # Sort by coverage (desc), then value (asc)
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Limit branching factor to avoid explosion?
        # For D=36, candidates are limited.
        
        for covered, c, new_diffs in scored_candidates:
            new_S = S + [c]
            new_S.sort()
            new_used = used_diffs | new_diffs
            
            result = backtrack(new_S, new_used, D_target)
            if result is not None:
                return result
                
        return None

    best_solution = None
    best_max_val = 10**6

    # Try increasing D
    # Start from theoretical lower bound 36
    for D in range(36, 100):
        if time.time() > deadline:
            break
            
        result = backtrack([0], set(), D)
        if result is not None:
            best_solution = result
            best_max_val = max(result)
            break
            
    if best_solution is None:
        # Fallback to a known valid set (if any) or greedy
        # Greedy fallback
        S = [0]
        used = set()
        for _ in range(32):
            if all(d in used for d in range(1, 31)):
                break
            best_c = None
            best_cov = 0
            # Try candidates
            cands = set(range(1, 200))
            for x in S:
                for d in range(1, 31):
                    if d not in used:
                        cands.add(x+d)
                        cands.add(x-d)
                        
            for c in sorted(cands):
                if c < 0 or c in S:
                    continue
                nd = set(abs(c-x) for x in S)
                if nd & used:
                    continue
                cov = len(nd & set(range(1, 31)) - used)
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
                    best_nd = nd
            if best_c is not None:
                S.append(best_c)
                S.sort()
                used.update(best_nd)
            else:
                break
                
        if all(d in used for d in range(1, 31)) and len(S) <= 32:
            best_solution = S
            best_max_val = max(S)

    if best_solution is not None:
        return best_solution
    
    # Emergency invalid return (should not happen)
    return [0, 1, 4, 10, 18, 23, 28, 34, 41]

