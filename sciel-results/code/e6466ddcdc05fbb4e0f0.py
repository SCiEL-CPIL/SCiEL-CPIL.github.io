"""B2[2]-set construction for {1..1000} with conflict-graph scoring and simulated annealing."""
import time
import random
import math

N = 1000
TIME_LIMIT = 115  # seconds; leaves margin before the 150 s kill

def _count_sums(selected):
    """Return a dict mapping each sum to its count, for the given selected set."""
    counts = {}
    s = sorted(selected)
    for i in range(len(s)):
        a = s[i]
        sa = 2 * a
        counts[sa] = counts.get(sa, 0) + 1
        for j in range(i + 1, len(s)):
            b = s[j]
            s2 = a + b
            counts[s2] = counts.get(s2, 0) + 1
    return counts

def _is_valid(counts):
    """Check if all sums have at most 2 representations."""
    for c in counts.values():
        if c > 2:
            return False
    return True

def _conflict_score(x, chosen, counts):
    """
    Score how many conflicts x would create if added to chosen.
    Returns (num_conflicts, max_overcount) where num_conflicts is the number of sums
    that would exceed 2, and max_overcount is the worst overcount.
    """
    num_conflicts = 0
    max_overcount = 0
    # Check 2*x
    c = counts.get(2 * x, 0) + 1
    if c > 2:
        num_conflicts += 1
        max_overcount = max(max_overcount, c - 2)
    # Check x + y for all y in chosen
    for y in chosen:
        s2 = x + y
        c = counts.get(s2, 0) + 1
        if c > 2:
            num_conflicts += 1
            max_overcount = max(max_overcount, c - 2)
    return num_conflicts, max_overcount

def _greedy_build_conflict_aware(order=None, time_limit=100.0):
    """
    Greedy construction with conflict-aware scoring.
    At each step, pick the candidate with the fewest conflicts.
    """
    start_time = time.time()
    if order is None:
        order = list(range(1, N + 1))
    
    chosen = []
    chosen_set = set()
    counts = {}
    remaining = set(order)
    
    while remaining and (time.time() - start_time) < time_limit:
        best_x = None
        best_conflicts = float('inf')
        best_overcount = float('inf')
        
        # Sample a subset if remaining is large to save time
        if len(remaining) > 200:
            sample = random.sample(list(remaining), 200)
        else:
            sample = list(remaining)
        
        for x in sample:
            conflicts, overcount = _conflict_score(x, chosen, counts)
            # Prefer fewer conflicts, then smaller overcount, then smaller x
            if (conflicts, overcount, x) < (best_conflicts, best_overcount, N + 1):
                best_conflicts = conflicts
                best_overcount = overcount
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        chosen.append(best_x)
        chosen_set.add(best_x)
        remaining.discard(best_x)
        
        # Update counts
        counts[2 * best_x] = counts.get(2 * best_x, 0) + 1
        for y in chosen:
            if y == best_x:
                continue
            s2 = best_x + y
            counts[s2] = counts.get(s2, 0) + 1
    
    return sorted(chosen)

def _try_local_search_sa(initial_set, time_limit, seed=42):
    """
    Simulated annealing local search: swap elements, accept worse states with probability.
    Objective: maximize size (primary), then minimize total overcount (secondary).
    """
    best = sorted(initial_set)
    best_counts = _count_sums(best)
    best_size = len(best)
    chosen_set = set(best)
    start_time = time.time()
    rng = random.Random(seed)
    
    candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    rng.shuffle(candidates)
    
    # Current state
    current = list(best)
    current_counts = dict(best_counts)
    current_size = len(current)
    current_set = set(current)
    
    # Compute a "badness" score: number of sums with count > 2, weighted by overcount
    def _badness(counts):
        bad = 0
        for c in counts.values():
            if c > 2:
                bad += (c - 2)
        return bad
    
    current_badness = _badness(current_counts)
    
    # Annealing schedule
    t_start = time.time()
    total_time = time_limit
    T0 = 10.0  # initial temperature
    T_end = 0.01
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        elapsed = time.time() - t_start
        progress = min(1.0, elapsed / total_time)
        T = T0 * math.exp(-5 * progress)  # exponential cooling
        
        # Pick a random swap
        if not candidates:
            break
        new_elem = rng.choice(candidates)
        if new_elem in current_set:
            continue
        i = rng.randrange(len(current))
        old_elem = current[i]
        
        # Compute trial state
        trial_counts = dict(current_counts)
        
        # Remove old_elem's contributions
        for y in current:
            if y == old_elem:
                continue
            s2 = old_elem + y
            if s2 in trial_counts:
                trial_counts[s2] -= 1
                if trial_counts[s2] < 0:
                    trial_counts[s2] = 0
        s2 = 2 * old_elem
        if s2 in trial_counts:
            trial_counts[s2] -= 1
            if trial_counts[s2] < 0:
                trial_counts[s2] = 0
        
        # Add new_elem's contributions
        for y in current:
            if y == old_elem:
                continue
            s2 = new_elem + y
            trial_counts[s2] = trial_counts.get(s2, 0) + 1
        s2 = 2 * new_elem
        trial_counts[s2] = trial_counts.get(s2, 0) + 1
        
        trial_badness = _badness(trial_counts)
        trial_size = len(current)  # size doesn't change with swap
        
        # Accept if valid and same size, or with probability based on badness change
        if trial_badness <= current_badness:
            # Accept: valid or less bad
            current[i] = new_elem
            current.sort()
            current_counts = trial_counts
            current_badness = trial_badness
            current_set = set(current)
            if new_elem in candidates:
                candidates.remove(new_elem)
            improved = True
            
            # Track best
            if current_size > best_size or (current_size == best_size and trial_badness < _badness(best_counts)):
                best = list(current)
                best_counts = dict(trial_counts)
                best_size = current_size
        else:
            # Metropolis acceptance: accept worse with probability exp(-delta/T)
            delta = trial_badness - current_badness
            if T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10)):
                current[i] = new_elem
                current.sort()
                current_counts = trial_counts
                current_badness = trial_badness
                current_set = set(current)
                if new_elem in candidates:
                    candidates.remove(new_elem)
                improved = True
    
    # Return the best valid state found
    if _is_valid(best_counts):
        return best
    # Fallback: return current if valid, else best
    if _is_valid(current_counts):
        return current
    return best

def _try_local_search(initial_set, time_limit, seed=42):
    """
    Try to improve the set via local search: swap out an element for another.
    Returns the best set found.
    """
    best = sorted(initial_set)
    best_counts = _count_sums(best)
    best_size = len(best)
    chosen_set = set(best)
    start_time = time.time()
    rng = random.Random(seed)
    
    # Candidate pool: all numbers not in best
    candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    rng.shuffle(candidates)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle candidates occasionally to escape local patterns
        if rng.random() < 0.1:
            rng.shuffle(candidates)
            
        for new_elem in candidates:
            if new_elem in chosen_set:
                continue
            # Try swapping new_elem in for each element in best
            for i in range(len(best)):
                old_elem = best[i]
                
                # Quick check: compute new counts incrementally
                trial = dict(best_counts)
                
                # Subtract old_elem's contributions
                for y in best:
                    if y == old_elem:
                        continue
                    s2 = old_elem + y
                    if s2 in trial:
                        trial[s2] -= 1
                        if trial[s2] < 0:
                            trial[s2] = 0
                s2 = 2 * old_elem
                if s2 in trial:
                    trial[s2] -= 1
                    if trial[s2] < 0:
                        trial[s2] = 0
                
                # Add new_elem's contributions
                for y in best:
                    if y == old_elem:
                        continue
                    s2 = new_elem + y
                    trial[s2] = trial.get(s2, 0) + 1
                s2 = 2 * new_elem
                trial[s2] = trial.get(s2, 0) + 1
                
                # Check validity
                valid = True
                for c in trial.values():
                    if c > 2:
                        valid = False
                        break
                    
                if valid:
                    # Accept the swap
                    best[i] = new_elem
                    best.sort()
                    best_counts = trial
                    chosen_set = set(best)
                    if new_elem in candidates:
                        candidates.remove(new_elem)
                    improved = True
                    break
                
                if (time.time() - start_time) > time_limit:
                    break
            if (time.time() - start_time) > time_limit:
                break
    
    return best

def _fill_remaining(best_set, counts, time_limit):
    """
    Try to add remaining elements in descending order (conflict-aware filling).
    Larger numbers tend to have fewer conflicts with small numbers already present.
    """
    chosen = sorted(best_set)
    chosen_set = set(chosen)
    start_time = time.time()
    
    # Try to add remaining values in descending order
    for x in range(N, 0, -1):
        if (time.time() - start_time) > time_limit:
            break
        if x in chosen_set:
            continue
        
        # Check if adding x keeps validity
        trial = dict(counts)
        trial[2 * x] = trial.get(2 * x, 0) + 1
        valid = True
        if trial[2 * x] > 2:
            valid = False
        else:
            for y in chosen:
                s2 = x + y
                trial[s2] = trial.get(s2, 0) + 1
                if trial[s2] > 2:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts = trial
    
    return chosen, counts

def construct():
    start_time = time.time()
    
    # Step 1: Multiple greedy constructions with different strategies
    best = []
    best_size = 0
    
    # Try increasing order (simple greedy)
    greedy1 = _greedy_build_conflict_aware(order=list(range(1, N + 1)), time_limit=10)
    if len(greedy1) > best_size:
        best = greedy1
        best_size = len(greedy1)
    
    # Try decreasing order
    if (time.time() - start_time) < TIME_LIMIT - 30:
        greedy2 = _greedy_build_conflict_aware(order=list(range(N, 0, -1)), time_limit=10)
        if len(greedy2) > best_size:
            best = greedy2
            best_size = len(greedy2)
    
    # Try random orderings with conflict-aware scoring
    rng = random.Random(123)
    for _ in range(8):
        if (time.time() - start_time) > TIME_LIMIT - 40:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        greedy_rand = _greedy_build_conflict_aware(order=order, time_limit=5)
        if len(greedy_rand) > best_size:
            best = greedy_rand
            best_size = len(greedy_rand)
    
    if (time.time() - start_time) > TIME_LIMIT - 20:
        return best
    
    # Step 2: Simulated annealing local search to improve via swaps
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 15:
        improved = _try_local_search_sa(best, remaining_time, seed=42)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if (time.time() - start_time) > TIME_LIMIT - 10:
        return best
    
    # Step 3: Fill remaining with conflict-aware descending pass
    counts = _count_sums(best)
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 3
    if remaining_time > 5:
        filled, counts = _fill_remaining(best, counts, remaining_time)
        if len(filled) > best_size:
            best = filled
            best_size = len(best)
    
    # Step 4: One more local search pass if time remains
    if (time.time() - start_time) < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - (time.time() - start_time) - 2
        if remaining_time > 5:
            improved2 = _try_local_search(best, remaining_time, seed=99)
            if len(improved2) > best_size:
                best = improved2
                best_size = len(best)
    
    return best
