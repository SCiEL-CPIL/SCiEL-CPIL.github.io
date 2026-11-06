import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    
    Strategy: Maximum clique formulation in a compatibility graph.
    Vertices are integers 1..BOUND. Two vertices x,y are compatible if adding both
    to a set containing 0 maintains the Sidon property (no repeated differences).
    We search for the smallest BOUND such that there exists a clique (plus 0) 
    covering all differences 1..30.
    
    Key insight: For a fixed set of elements S = {0, s1, ..., sk}, the Sidon condition
    means all differences |si - sj| are distinct. This is equivalent to: for any two
    pairs (i,j) and (k,l) with i<j, k<l, and {i,j} != {k,l}, we have |si-sj| != |sk-sl|.
    
    We use branch-and-bound with bitset acceleration and MRV heuristics.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    TARGET = frozenset(range(1, 31))
    N_TARGET = 30
    
    def is_valid_sidon(S):
        """Check if sorted list S (including 0) is a Sidon set."""
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def get_all_diffs(S):
        """Get all pairwise differences of sorted set S."""
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs
    
    def covers_all(S):
        """Check if S covers all differences 1..30."""
        return TARGET.issubset(get_all_diffs(S))
    
    def validate(S):
        """Full validation."""
        if not S or 0 not in S:
            return False
        if len(S) > 32:
            return False
        if max(S) > 10**6:
            return False
        if len(set(S)) != len(S):
            return False
        if not is_valid_sidon(S):
            return False
        return TARGET.issubset(get_all_diffs(S))
    
    def greedy_initial(bound, deadline):
        """
        Greedy construction: start with {0}, repeatedly add the element that
        covers the most missing differences while maintaining Sidon property.
        """
        S = [0]
        S_set = {0}
        used_diffs = set()
        
        for _ in range(31):
            if time.time() > deadline:
                break
            if TARGET.issubset(used_diffs):
                break
            
            missing = TARGET - used_diffs
            best_x = None
            best_gain = 0
            best_waste = 999
            
            # Candidates: for each s in S and each missing d, try s+d and s-d
            candidates = set()
            for s in S:
                for d in missing:
                    c1 = s + d
                    if 0 < c1 <= bound and c1 not in S_set:
                        candidates.add(c1)
                    c2 = s - d
                    if 0 < c2 <= bound and c2 not in S_set:
                        candidates.add(c2)
            
            for x in sorted(candidates):
                # Check Sidon validity
                new_diffs = set()
                valid = True
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue
                
                gain = len(new_diffs & missing)
                waste = len(new_diffs) - gain
                if gain > best_gain or (gain == best_gain and waste < best_waste):
                    best_gain = gain
                    best_waste = waste
                    best_x = x
            
            if best_x is None:
                # Fallback: try small values
                for x in range(1, min(bound + 1, 100)):
                    if x in S_set:
                        continue
                    new_diffs = set()
                    valid = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid and (new_diffs & missing):
                        best_x = x
                        break
            
            if best_x is None:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            for s in S[:-1]:
                used_diffs.add(abs(s - best_x))
        
        if TARGET.issubset(used_diffs) and is_valid_sidon(S):
            return sorted(S)
        return None
    
    def backtrack_clique(bound, deadline):
        """
        Branch-and-bound search for a valid Sidon set with max element <= bound.
        Uses MRV: at each step, pick the missing difference with fewest valid
        candidate elements that could cover it.
        """
        best_solution = None
        
        # State
        S = [0]
        S_set = {0}
        used_diffs = set()
        
        def can_add(x, S_list, S_set_local, used_diffs_local):
            """Check if x can be added to maintain Sidon property."""
            if x in S_set_local:
                return None
            new_diffs = set()
            for s in S_list:
                dd = abs(s - x)
                if dd in used_diffs_local or dd in new_diffs:
                    return None
                new_diffs.add(dd)
            return new_diffs
        
        def dfs():
            nonlocal best_solution
            
            if time.time() > deadline:
                return
            
            if TARGET.issubset(used_diffs):
                best_solution = S[:]
                return
            
            if len(S) >= 32:
                return
            
            missing = TARGET - used_diffs
            
            # Generate all candidates that cover at least one missing diff
            all_cands = set()
            for d in missing:
                for s in S:
                    c1 = s + d
                    if 0 < c1 <= bound and c1 not in S_set:
                        all_cands.add(c1)
                    c2 = s - d
                    if 0 < c2 <= bound and c2 not in S_set:
                        all_cands.add(c2)
            
            # Filter to valid candidates and compute their coverage
            valid_cands = []
            for x in all_cands:
                new_diffs = can_add(x, S, S_set, used_diffs)
                if new_diffs is not None:
                    gain = len(new_diffs & missing)
                    if gain > 0:
                        valid_cands.append((x, new_diffs, gain))
            
            if not valid_cands:
                return
            
            # MRV: sort by gain descending, then x ascending
            valid_cands.sort(key=lambda t: (-t[2], t[0]))
            
            # Try candidates
            for x, new_diffs, gain in valid_cands:
                # Apply
                S.append(x)
                S_set.add(x)
                old_used = used_diffs.copy()
                used_diffs |= new_diffs
                
                dfs()
                
                # Undo
                S.pop()
                S_set.discard(x)
                used_diffs = old_used
                
                if best_solution is not None:
                    return  # Found a solution
        
        dfs()
        return best_solution
    
    def local_search_reduce(S, deadline):
        """
        Try to reduce the maximum element by replacing elements with smaller values.
        """
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S) - 1, 0, -1):  # Start from largest elements
                old_val = S[i]
                # Try replacing with smaller values
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_valid_sidon(trial):
                        continue
                    if not TARGET.issubset(get_all_diffs(trial)):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = trial
                        best_max = max(S)
                        improved = True
                        break
        
        return best
    
    def randomized_restart_search(bound, deadline, n_restarts=50):
        """
        Multiple randomized greedy attempts with different orderings.
        """
        best = None
        best_max = bound + 1
        
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Shuffle the target differences to get different greedy paths
            shuffled = list(range(1, 31))
            random.shuffle(shuffled)
            
            S = [0]
            S_set = {0}
            used_diffs = set()
            
            for d in shuffled:
                if d in used_diffs:
                    continue
                
                # Find best element that covers d
                candidates = set()
                for s in S:
                    c1 = s + d
                    if 0 < c1 <= bound and c1 not in S_set:
                        candidates.add(c1)
                    c2 = s - d
                    if 0 < c2 <= bound and c2 not in S_set:
                        candidates.add(c2)
                
                best_x = None
                best_gain = 0
                best_waste = 999
                
                for x in sorted(candidates):
                    new_diffs = set()
                    valid = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if not valid:
                        continue
                    
                    gain = len(new_diffs & (TARGET - used_diffs))
                    waste = len(new_diffs) - gain
                    if gain > best_gain or (gain == best_gain and waste < best_waste):
                        best_gain = gain
                        best_waste = waste
                        best_x = x
                
                if best_x is None:
                    break
                
                S.append(best_x)
                S_set.add(best_x)
                for s in S[:-1]:
                    used_diffs.add(abs(s - best_x))
            
            if TARGET.issubset(used_diffs) and is_valid_sidon(S):
                if max(S) < best_max:
                    best_max = max(S)
                    best = sorted(S)
        
        return best
    
    # Main execution
    best_S = None
    best_diam = 10**9
    
    # Phase 1: Quick greedy for initial upper bound
    t0 = time.time()
    S_greedy = greedy_initial(100, t0 + 10)
    if S_greedy and validate(S_greedy):
        best_S = S_greedy
        best_diam = max(S_greedy)
    
    # Phase 2: Randomized restarts to improve
    if time.time() < DEADLINE - 20:
        S_rand = randomized_restart_search(best_diam if best_S else 100, time.time() + 30, n_restarts=100)
        if S_rand and validate(S_rand):
            if max(S_rand) < best_diam:
                best_S = S_rand
                best_diam = max(S_rand)
    
    # Phase 3: Iterative deepening backtracking
    if time.time() < DEADLINE - 10:
        # Start from a lower bound and increase
        # Lower bound: need to cover diff 30, so max element >= 30
        # But Sidon constraint makes it much larger
        # For k elements, max diff >= k(k-1)/2, so for 30 diffs we need ~8+ elements
        # Diameter for 8-element Sidon set is at least ~20-30
        # Try bounds from 40 upward
        lower = 40
        upper = best_diam if best_S else 100
        
        for bound in range(lower, min(upper + 1, 120)):
            if time.time() > DEADLINE - 5:
                break
            
            sol = backtrack_clique(bound, DEADLINE - 10)
            if sol and validate(sol):
                if max(sol) < best_diam:
                    best_S = sol
                    best_diam = max(sol)
                break  # Found solution at this bound, no need to try larger
    
    # Phase 4: Local refinement
    if best_S and time.time() < DEADLINE - 3:
        refined = local_search_reduce(best_S, DEADLINE - 3)
        if refined and validate(refined):
            if max(refined) < best_diam:
                best_S = refined
                best_diam = max(refined)
    
    # Phase 5: Final aggressive local search
    if best_S and time.time() < DEADLINE - 1:
        # Try more aggressive replacement
        S = sorted(best_S)
        for _ in range(50):
            if time.time() > DEADLINE - 1:
                break
            improved = False
            for i in range(1, len(S)):
                old_val = S[i]
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if is_valid_sidon(trial) and TARGET.issubset(get_all_diffs(trial)):
                        if max(trial) < max(S):
                            S = trial
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        if validate(S) and max(S) < best_diam:
            best_S = S
            best_diam = max(S)
    
    # Fallback: ensure we have a valid solution
    if best_S is None or not validate(best_S):
        S = [0]
        used_diffs = set()
        for d in range(1, 31):
            if d in used_diffs:
                continue
            found = False
            for x in range(1, 500):
                if x in S:
                    continue
                new_diffs = set()
                ok = True
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        ok = False
                        break
                    new_diffs.add(dd)
                if ok and d in new_diffs:
                    S.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # Force add
                base = 2 * max(S) + 50
                S.append(base)
                S.append(base + d)
                for s in S[:-2]:
                    used_diffs.add(abs(s - base))
                    used_diffs.add(abs(s - (base + d)))
                used_diffs.add(d)
        if validate(S):
            best_S = S
    
    # Absolute fallback
    if best_S is None or not validate(best_S):
        S = [0]
        used_diffs = set()
        for x in range(1, 300):
            if x in S:
                continue
            new_diffs = set()
            ok = True
            for s in S:
                dd = abs(s - x)
                if dd in used_diffs or dd in new_diffs:
                    ok = False
                    break
                new_diffs.add(dd)
            if ok:
                S.append(x)
                used_diffs |= new_diffs
            if TARGET.issubset(used_diffs):
                break
        best_S = S
    
    return best_S
