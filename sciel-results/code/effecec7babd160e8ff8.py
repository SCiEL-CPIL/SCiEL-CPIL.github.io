"""Sidon set containing 0 with differences 1..30, minimizing the largest element."""
import time
import random
import sys

TIME_LIMIT = 145  # wall-clock seconds, safety margin under 150s

def construct():
    t0 = time.time()
    DEADLINE = t0 + TIME_LIMIT
    
    target_diffs = set(range(1, 31))
    
    def check_time():
        return time.time() < DEADLINE
    
    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def covers_target(S):
        """Check if all target differences 1..30 are present in S."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return target_diffs.issubset(diffs)
    
    def get_diffs(S):
        """Get all positive differences in S."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs
    
    def missing_diffs(S):
        """Get target differences not present in S."""
        return target_diffs - get_diffs(S)
    
    def greedy_construct(seed=0):
        """Greedy construction of a Sidon set covering 1..30."""
        random.seed(seed)
        S = [0]
        for d in range(1, 31):
            if d in get_diffs(S):
                continue
            # Try to find x such that adding x creates difference d
            # and keeps S Sidon
            candidates = set()
            for y in S:
                if y + d >= 0:
                    candidates.add(y + d)
                if y - d >= 0:
                    candidates.add(y - d)
            candidates = sorted([c for c in candidates if c not in S])
            for x in candidates:
                trial = sorted(S + [x])
                if is_sidon(trial):
                    S = trial
                    break
            else:
                # Fallback: find some x that doesn't break Sidon
                for x in range(1, 200):
                    if x in S:
                        continue
                    trial = sorted(S + [x])
                    if is_sidon(trial):
                        S = trial
                        break
                else:
                    # Very aggressive fallback
                    x = 2 * max(S) + 3 * d + 1
                    while True:
                        trial = sorted(S + [x])
                        if is_sidon(trial):
                            S = trial
                            break
                        x += 1
        return S
    
    def local_search(S, time_budget=20):
        """Local search to minimize the largest element while maintaining Sidon and coverage."""
        if not check_time():
            return S
        
        best_S = sorted(S)
        best_max = max(best_S)
        t_start = time.time()
        
        # Try to remove elements and re-add them to reduce diameter
        # Or try to swap elements for smaller ones
        
        # Strategy: iteratively try to replace each element with a smaller one
        # that maintains Sidon property and coverage
        
        for iteration in range(1000):
            if time.time() - t_start > time_budget or not check_time():
                break
            
            improved = False
            S_sorted = sorted(best_S)
            
            # Try to reduce each element
            for i in range(len(S_sorted) - 1, 0, -1):  # Skip index 0 (which is 0)
                old_val = S_sorted[i]
                # Try replacing with smaller values
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in best_S:
                        continue
                    trial = sorted([x if x != old_val else new_val for x in best_S])
                    if is_sidon(trial) and covers_target(trial):
                        new_max = max(trial)
                        if new_max < best_max:
                            best_S = trial
                            best_max = new_max
                            improved = True
                            break
            
            if not improved:
                # Try a random perturbation
                if random.random() < 0.3:
                    i = random.randint(1, len(best_S) - 1)
                    delta = random.randint(-5, 5)
                    new_val = best_S[i] + delta
                    if new_val > 0 and new_val not in best_S:
                        trial = sorted([x if x != best_S[i] else new_val for x in best_S])
                        if is_sidon(trial) and covers_target(trial):
                            new_max = max(trial)
                            if new_max < best_max:
                                best_S = trial
                                best_max = new_max
                            else:
                                # Accept with some probability (simulated annealing)
                                if random.random() < 0.1:
                                    best_S = trial
                                    best_max = new_max
        
        return best_S
    
    def minimize_diameter(S, time_budget=20):
        """Try to minimize the diameter by removing redundant elements."""
        if not check_time():
            return S
        
        best_S = sorted(S)
        best_max = max(best_S)
        t_start = time.time()
        
        # Try removing each element (except 0) and see if coverage is maintained
        while True:
            if time.time() - t_start > time_budget or not check_time():
                break
            
            removed = False
            S_sorted = sorted(best_S)
            
            for i in range(len(S_sorted) - 1, 0, -1):
                trial = S_sorted[:i] + S_sorted[i+1:]
                if covers_target(trial):
                    best_S = trial
                    best_max = max(best_S)
                    removed = True
                    break
            
            if not removed:
                break
        
        return best_S
    
    # Phase 1: Greedy construction with multiple seeds
    best_solution = None
    best_diameter = float('inf')
    
    for seed in range(20):
        if not check_time():
            break
        S = greedy_construct(seed)
        if S and is_sidon(S) and covers_target(S):
            if max(S) < best_diameter:
                best_solution = S[:]
                best_diameter = max(S)
    
    # Phase 2: Local search to improve
    if best_solution:
        S2 = local_search(best_solution, time_budget=30)
        if is_sidon(S2) and covers_target(S2):
            if max(S2) < best_diameter:
                best_solution = S2[:]
                best_diameter = max(S2)
    
    # Phase 3: Minimize diameter by removing redundant elements
    if best_solution:
        S3 = minimize_diameter(best_solution, time_budget=15)
        if is_sidon(S3) and covers_target(S3):
            if max(S3) < best_diameter:
                best_solution = S3[:]
                best_diameter = max(S3)
    
    # Additional runs with different random seeds
    for seed in [100, 200, 300, 400, 500]:
        if not check_time():
            break
        S = greedy_construct(seed)
        if S and is_sidon(S) and covers_target(S):
            S = local_search(S, time_budget=10)
            if is_sidon(S) and covers_target(S):
                S = minimize_diameter(S, time_budget=5)
                if is_sidon(S) and covers_target(S):
                    if max(S) < best_diameter:
                        best_solution = S[:]
                        best_diameter = max(S)
    
    if best_solution is not None:
        return sorted(best_solution)
    
    # Final fallback: simple greedy
    a = [0]
    for d in range(1, 31):
        diffs_set = get_diffs(a)
        if d in diffs_set:
            continue
        found = False
        for x in sorted(set(v + dd for v in a for dd in (d,) if v + dd >= 0) - set(a)):
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a
