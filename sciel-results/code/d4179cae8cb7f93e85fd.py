"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # We need a Sidon set containing 0, with all differences in 1..30 covered,
    # all positive differences distinct, at most 32 elements, minimize max element.
    
    # Strategy: Systematic backtracking with coverage-based ordering and pruning.
    # Start with {0} and build up the set, ensuring Sidon property and coverage.
    
    best_solution = None
    best_max_val = float('inf')
    
    def get_diffs(s):
        """Get all positive differences from sorted set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if s is a Sidon set (all positive differences distinct)."""
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        return len(diffs) == len(set(diffs))
    
    def covers_all(s, target=30):
        """Check if all differences 1..target are covered."""
        diffs = get_diffs(s)
        return all(d in diffs for d in range(1, target + 1))
    
    def has_collision(s):
        """Check if any positive difference appears more than once."""
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        return len(diffs) != len(set(diffs))
    
    def backtracking(current_set, covered, max_val_bound):
        """Backtracking search with pruning."""
        nonlocal best_solution, best_max_val, start_time, DEADLINE
        
        if time.time() > DEADLINE:
            return
        
        # Prune: if current max already exceeds best known, stop
        if max(current_set) >= best_max_val:
            return
        
        # If we've covered all targets, record solution
        if len(covered) == 30:
            max_val = max(current_set)
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = list(current_set)
            return
        
        # If set is too large, prune
        if len(current_set) > 32:
            return
        
        # Determine which differences are still missing
        missing = [d for d in range(1, 31) if d not in covered]
        if not missing:
            return
        
        # Try to add a new element to cover missing differences
        # Candidate positions: for each existing element x and each missing diff d,
        # try x+d and x-d (if non-negative)
        candidates = set()
        for x in current_set:
            for d in missing:
                if x + d < max_val_bound:
                    candidates.add(x + d)
                if x - d >= 0:
                    candidates.add(x - d)
        
        # Also try small positive integers not yet in set
        for v in range(1, min(max_val_bound, 200)):
            if v not in current_set:
                candidates.add(v)
        
        # Sort candidates by how many new differences they cover (heuristic)
        def coverage_score(cand):
            if cand in current_set:
                return 0
            new_set = sorted(current_set + [cand])
            if not is_sidon(new_set):
                return -1
            new_diffs = get_diffs(new_set)
            new_covered = new_diffs & set(missing)
            return len(new_covered)
        
        # Filter and sort candidates
        valid_candidates = []
        for cand in candidates:
            if cand in current_set:
                continue
            if cand >= max_val_bound:
                continue
            new_set = sorted(current_set + [cand])
            if not is_sidon(new_set):
                continue
            new_diffs = get_diffs(new_set)
            new_covered = new_diffs & set(missing)
            if new_covered:
                valid_candidates.append((cand, len(new_covered)))
        
        # Sort by coverage (most first), then by value (smallest first)
        valid_candidates.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit branching factor for speed
        for cand, cov in valid_candidates[:15]:
            new_set = sorted(current_set + [cand])
            new_diffs = get_diffs(new_set)
            new_covered = new_diffs & set(missing)
            if not new_covered:
                continue
            backtracking(new_set, covered | new_covered, max_val_bound)
            if time.time() > DEADLINE:
                return
    
    # Initial call with {0}
    initial_covered = set()
    max_bound = 500  # initial upper bound on max element
    backtracking([0], initial_covered, max_bound)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy approach if backtracking didn't find anything in time
    a = [0]
    for d in range(1, 31):
        diffs = get_diffs(sorted(a))
        if d in diffs:
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
