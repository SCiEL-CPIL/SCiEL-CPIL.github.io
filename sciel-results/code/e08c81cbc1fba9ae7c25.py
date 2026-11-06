import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses exact cover (Algorithm X with Dancing Links) on a reduced candidate universe
    for small diameters, with greedy fallback for larger bounds.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Leave 15s margin before 150s kill

    TARGET = set(range(1, 31))

    def is_sidon_fast(S):
        """Check if S is a Sidon set using a set of differences."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def score_set(S):
        diffs = get_diffs(S)
        missing = len(TARGET - diffs)
        return (missing, max(S) if S else 0)

    def greedy_init(max_val):
        """Greedy construction to get a starting point."""
        S = [0]
        used_diffs = set()
        covered = set()
        target = TARGET

        for step in range(31):
            if time.time() > DEADLINE - 20:
                break
            if covered == target:
                break

            best_x = None
            best_gain = 0

            missing = target - covered
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)
            for _ in range(100):
                candidates.add(random.randint(1, max_val))

            candidates = [c for c in candidates if c > 0 and c not in set(S) and c <= max_val]
            candidates.sort()

            for x in candidates[:300]:
                valid = True
                new_diffs = set()
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue
                gain = len(new_diffs & target)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain >= len(missing):
                        break

            if best_x is None:
                for x in range(1, min(max_val + 1, 500)):
                    if x in set(S):
                        continue
                    valid = True
                    new_diffs = set()
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid:
                        best_x = x
                        break
                if best_x is None:
                    break

            S.append(best_x)
            for s in S[:-1]:
                used_diffs.add(abs(s - best_x))
            covered = used_diffs & target

        return S

    def local_refine(S, deadline):
        """Try to reduce max(S) by replacing elements with smaller values."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_sidon_fast(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = trial
                        best_max = max(S)
                        improved = True
                        break
            if not improved:
                break

        return best

    def exact_cover_search(max_val, deadline):
        """
        Use exact cover / Algorithm X to find a Sidon set {0} ∪ X where:
        - All elements in [1, max_val]
        - All differences 1..30 are covered
        - All differences are distinct (Sidon property)
        
        We model this as: choose a subset of {1,...,max_val} such that:
        1. For each d in 1..30: at least one pair (a,b) in {0}∪X has b-a=d
        2. For each d in 1..max_val: at most one pair has difference d
        
        This is a hitting set / exact cover problem. We use backtracking with
        constraint propagation.
        """
        if time.time() > deadline:
            return None

        # Precompute for each candidate x in 1..max_val:
        # - which target differences it can cover (with existing 0 and other candidates)
        # - which differences it creates (must all be distinct)
        
        # Strategy: backtracking that picks candidates in order, maintaining
        # the set of used differences and covered target differences.
        
        best_solution = None
        best_size = 33  # max 32 elements including 0, so max 31 more
        
        used_diffs = set()
        covered = set()
        current_set = [0]
        
        def backtrack(candidates, depth):
            nonlocal best_solution, best_size, used_diffs, covered, current_set
            
            if time.time() > deadline:
                return False
            
            if covered == TARGET:
                # Found a solution
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = current_set[:]
                return True
            
            if depth >= best_size - 1:
                return False
            
            if not candidates:
                return False
            
            # Choose the most constrained candidate: one that covers the most
            # missing differences while creating the fewest conflicts
            missing = TARGET - covered
            
            # For each candidate, compute:
            # - how many missing diffs it covers
            # - how many new diffs it creates
            # - whether all new diffs are unused
            best_cand = None
            best_score = -1
            best_new_diffs = None
            
            for x in candidates:
                # Check if x can be added (all diffs with current set are unused)
                new_diffs = set()
                valid = True
                for s in current_set:
                    dd = abs(s - x)
                    if dd in used_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue
                
                # Count how many missing target diffs this covers
                gain = len(new_diffs & missing)
                if gain == 0:
                    continue
                
                # Score: prefer candidates that cover more missing diffs
                # and create fewer total new diffs (less constraint)
                score = gain * 10 - len(new_diffs)
                if score > best_score:
                    best_score = score
                    best_cand = x
                    best_new_diffs = new_diffs
            
            if best_cand is None:
                # No candidate can help; try removing some constraint
                # or just fail this branch
                return False
            
            # Add best_cand
            old_used = set(used_diffs)
            old_covered = set(covered)
            old_current = current_set[:]
            
            current_set.append(best_cand)
            used_diffs = used_diffs | best_new_diffs
            covered = covered | (best_new_diffs & TARGET)
            
            # Remove best_cand and any conflicting candidates from remaining
            new_candidates = []
            for x in candidates:
                if x == best_cand:
                    continue
                # Check if x conflicts with best_cand
                conflict = False
                dd = abs(x - best_cand)
                if dd in used_diffs:
                    conflict = True
                if not conflict:
                    # Also check x against all of current_set
                    valid = True
                    for s in current_set:
                        if s == x:
                            continue
                        dd2 = abs(s - x)
                        if dd2 in used_diffs:
                            valid = False
                            break
                    if valid:
                        new_candidates.append(x)
            
            if backtrack(new_candidates, depth + 1):
                return True
            
            # Backtrack
            current_set = old_current
            used_diffs = old_used
            covered = old_covered
            
            return False
        
        candidates = list(range(1, max_val + 1))
        backtrack(candidates, 0)
        
        return best_solution

    # Main search
    best_S = None
    best_score = (30, 10**9)

    # First, get a valid solution with greedy
    random.seed(42)
    S0 = greedy_init(1000000)
    if S0 and covers_all(S0) and is_sidon_fast(S0):
        s0 = score_set(S0)
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Try exact cover search with decreasing max_val
    # Start from a reasonable upper bound and work down
    upper_bound = 100
    if best_S:
        upper_bound = max(upper_bound, max(best_S) + 1)
    
    for max_val in range(upper_bound, 30, -1):
        if time.time() > DEADLINE - 10:
            break
        
        sol = exact_cover_search(max_val, DEADLINE - 15)
        if sol and covers_all(sol) and is_sidon_fast(sol):
            s = score_set(sol)
            if s < best_score:
                best_score = s
                best_S = sol
                # Try local refinement
                if time.time() < DEADLINE - 10:
                    refined = local_refine(sol, DEADLINE - 10)
                    if refined and max(refined) < max(sol):
                        rs = score_set(refined)
                        if rs < best_score:
                            best_score = rs
                            best_S = refined
                break
    
    # If we have a best_S, try local refinement to reduce diameter
    if best_S and time.time() < DEADLINE - 5:
        refined = local_refine(best_S, DEADLINE - 5)
        if refined:
            rs = score_set(refined)
            if rs < best_score:
                best_score = rs
                best_S = refined

    # Final check
    if best_S is None or not covers_all(best_S) or not is_sidon_fast(best_S):
        # Fallback: simple construction
        best_S = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            for x in range(1, 1000000):
                if x in best_S:
                    continue
                trial = sorted(best_S + [x])
                if is_sidon_fast(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        best_S = trial
                        covered = trial_diffs & TARGET
                        break
            else:
                y = 2 * max(best_S) + d + 1
                best_S = sorted(best_S + [y, y + d])
                covered = get_diffs(best_S) & TARGET

    return best_S
