import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 150s kill
    
    TARGETS = frozenset(range(1, 31))
    
    def _compute_diffs(s):
        """Return (diff_set, collision_count, missing_count) for sorted list s."""
        n = len(s)
        diff_set = set()
        collisions = 0
        for i in range(n):
            si = s[i]
            for j in range(i + 1, n):
                d = s[j] - si
                if d in diff_set:
                    collisions += 1
                else:
                    diff_set.add(d)
        missing = len(TARGETS - diff_set)
        return diff_set, collisions, missing
    
    def _energy(s):
        """Energy = max(s) + 10000*collisions + 5000*missing."""
        if not s or 0 not in s:
            return 10**9
        diff_set, collisions, missing = _compute_diffs(s)
        return max(s) + 10000 * collisions + 5000 * missing
    
    def _is_valid(s):
        """Check if s is a valid Sidon set covering 1..30."""
        if not s or 0 not in s:
            return False
        if len(s) > 32:
            return False
        if max(s) > 10**6:
            return False
        diff_set, collisions, missing = _compute_diffs(s)
        return collisions == 0 and missing == 0
    
    def _greedy_init():
        """Build initial solution using greedy approach."""
        a = [0]
        current_diffs = set()
        
        for d in range(1, 31):
            if d in current_diffs:
                continue
            found = False
            # Try adding a single element that creates difference d
            candidates = set()
            for x in a:
                candidates.add(x + d)
                if x - d > 0:
                    candidates.add(x - d)
            for cand in sorted(candidates):
                if cand in a or cand < 0:
                    continue
                trial = sorted(a + [cand])
                diff_set, collisions, _ = _compute_diffs(trial)
                if collisions == 0:
                    a = trial
                    current_diffs = diff_set
                    found = True
                    break
            if not found:
                # Try adding two elements
                for y in range(1, 500):
                    for z in range(y + 1, y + 50):
                        trial = sorted(a + [y, z])
                        diff_set, collisions, _ = _compute_diffs(trial)
                        if collisions == 0 and d in diff_set:
                            a = trial
                            current_diffs = diff_set
                            found = True
                            break
                    if found:
                        break
            if not found:
                # Fallback: add large elements
                y = 2 * max(a) + d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    diff_set, collisions, _ = _compute_diffs(trial)
                    if collisions == 0:
                        a = trial
                        current_diffs = diff_set
                        break
                    y += 1
        return a
    
    def _simulated_annealing(initial, time_limit):
        """Fast simulated annealing to minimize max element."""
        best = list(initial)
        best_e = _energy(best)
        current = list(best)
        current_e = best_e
        
        T = 500.0
        T_min = 0.1
        cooling = 0.99999
        
        # Precompute for speed
        while time.time() < time_limit:
            T = max(T_min, T * cooling)
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace an element with a nearby value
                if len(current) <= 1:
                    continue
                idx = random.randint(1, len(current) - 1)
                old_val = current[idx]
                if old_val <= 1:
                    continue
                # Try a nearby value (smaller preferred)
                delta_val = random.randint(-max(1, old_val // 4), max(1, old_val // 4))
                new_val = old_val + delta_val
                if new_val <= 0 or new_val == old_val:
                    continue
                if new_val in current:
                    continue
                trial = current.copy()
                trial[idx] = new_val
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif move_type < 0.8:
                # Swap two elements
                if len(current) <= 2:
                    continue
                i, j = random.sample(range(1, len(current)), 2)
                trial = current.copy()
                trial[i], trial[j] = trial[j], trial[i]
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif move_type < 0.9:
                # Remove an element
                if len(current) <= 6:
                    continue
                idx = random.randint(1, len(current) - 1)
                trial = current[:idx] + current[idx + 1:]
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            else:
                # Add an element
                if len(current) >= 32:
                    continue
                max_so_far = max(current)
                new_val = random.randint(1, max_so_far + 30)
                if new_val in current:
                    continue
                trial = sorted(current + [new_val])
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
        
        return best
    
    random.seed(42)
    
    # Phase 1: Generate initial solution
    initial = _greedy_init()
    
    # Phase 2: Simulated annealing to refine
    time_limit = DEADLINE - 5.0
    result = _simulated_annealing(initial, time_limit)
    
    # Ensure validity - if not valid, try to fix or fall back
    if not _is_valid(result):
        # Try to fix by removing problematic elements
        fixed = list(result)
        for _ in range(100):
            if time.time() > DEADLINE - 2:
                break
            if _is_valid(fixed):
                break
            diff_set, collisions, missing = _compute_diffs(fixed)
            if collisions > 0:
                # Remove a random non-zero element
                if len(fixed) > 1:
                    idx = random.randint(1, len(fixed) - 1)
                    fixed = fixed[:idx] + fixed[idx + 1:]
            elif missing > 0:
                # Add elements to cover missing differences
                missing_diffs = TARGETS - diff_set
                if missing_diffs and len(fixed) < 32:
                    d = random.choice(list(missing_diffs))
                    for x in fixed:
                        cand = x + d
                        if cand > 0 and cand not in fixed:
                            trial = sorted(fixed + [cand])
                            _, coll, _ = _compute_diffs(trial)
                            if coll == 0:
                                fixed = trial
                                break
        result = fixed
    
    # Final validation and cleanup
    result = sorted(set(result))
    if 0 not in result:
        result = sorted([0] + result)
    
    # If still invalid, return a known valid solution
    if not _is_valid(result):
        # Return the initial greedy solution which should be valid
        result = initial
    
    return result
