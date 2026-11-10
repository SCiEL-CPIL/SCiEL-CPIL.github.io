import time
import sys
import random
import math

def _all_block_sums(seq):
    """Return the set of all nonempty consecutive-block sums, or None if collision."""
    n = len(seq)
    if n == 0:
        return set()
    # Prefix sums for O(1) range-sum queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + seq[i]
    seen = set()
    for i in range(n):
        pi = prefix[i]
        for j in range(i + 1, n + 1):
            s = prefix[j] - pi
            if s in seen:
                return None
            seen.add(s)
    return seen

def _is_valid(seq):
    if not seq:
        return True
    # Strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if seq[0] < 1 or seq[-1] > 256:
        return False
    return _all_block_sums(seq) is not None

def _greedy_construct(min_val=1, max_val=256):
    """Greedily build a valid sequence, trying smallest feasible next value."""
    seq = []
    for x in range(min_val, max_val + 1):
        # Try appending x
        trial = seq + [x]
        if _is_valid(trial):
            seq = trial
    return seq

def _dfs_extend(seq, max_val, deadline):
    """DFS with backtracking to extend a sequence. Returns best found."""
    best = list(seq)
    
    def dfs(current, seen_sums, last_val):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) > len(best):
            best = list(current)
        # Try to extend
        for x in range(last_val + 1, max_val + 1):
            if time.time() > deadline:
                return
            # Check new block sums ending at x
            ok = True
            s = 0
            # Iterate backwards from x to current[0]
            for k in range(len(current) - 1, -1, -1):
                s += current[k] if k == len(current) - 1 else (current[k] + (s - current[k]))
                # Actually simpler: just accumulate
            # Redo: accumulate from x backwards
            s = x
            if s in seen_sums:
                continue
            for k in range(len(current) - 1, -1, -1):
                s += current[k]
                if s in seen_sums:
                    ok = False
                    break
            if not ok:
                continue
            # All new sums are valid; add them to seen_sums
            new_sums = [x]
            s = x
            for k in range(len(current) - 1, -1, -1):
                s += current[k]
                new_sums.append(s)
            for ns in new_sums:
                seen_sums.add(ns)
            current.append(x)
            dfs(current, seen_sums, x)
            current.pop()
            for ns in new_sums:
                seen_sums.discard(ns)
    
    # Precompute seen_sums for initial seq
    seen = _all_block_sums(seq)
    if seen is None:
        return best
    dfs(list(seq), seen, seq[-1] if seq else 0)
    return best

def _local_search(seq, time_limit, rng):
    """Local search over the gap vector to maximize length."""
    best = list(seq)
    best_len = len(seq)
    n = len(best)
    
    def rebuild(start, gaps):
        s = [start]
        for g in gaps:
            s.append(s[-1] + g)
        return s
    
    # Represent as (start, gaps)
    start = best[0]
    gaps = [best[i+1] - best[i] for i in range(n-1)]
    
    deadline = time_limit
    temp = 10.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Try mutations
        moved = False
        for _ in range(500):
            if time.time() >= deadline:
                break
            trial_start = start
            trial_gaps = list(gaps)
            
            op = rng.random()
            if op < 0.3:
                # Perturb a gap
                idx = rng.randrange(len(trial_gaps))
                delta = rng.randint(-3, 3)
                trial_gaps[idx] += delta
                if trial_gaps[idx] <= 0:
                    continue
            elif op < 0.5:
                # Change start
                trial_start += rng.randint(-2, 2)
                if trial_start < 1:
                    continue
            elif op < 0.7:
                # Try to add a gap (extend sequence)
                new_gap = rng.randint(1, 8)
                trial_gaps.append(new_gap)
                last = trial_start + sum(trial_gaps)
                if last > 256:
                    continue
            elif op < 0.85 and len(trial_gaps) > 1:
                # Remove a gap (shorten)
                idx = rng.randrange(len(trial_gaps))
                trial_gaps.pop(idx)
            else:
                # Swap two adjacent gaps
                if len(trial_gaps) < 2:
                    continue
                i = rng.randrange(len(trial_gaps) - 1)
                trial_gaps[i], trial_gaps[i+1] = trial_gaps[i+1], trial_gaps[i]
            
            trial = rebuild(trial_start, trial_gaps)
            if not _is_valid(trial):
                continue
            
            if len(trial) > best_len:
                best = trial
                best_len = len(trial)
                start = trial_start
                gaps = trial_gaps
                temp = 10.0
                moved = True
            else:
                # Accept with some probability (simulated annealing)
                if rng.random() < math.exp(-0.5 / max(temp, 0.1)):
                    start = trial_start
                    gaps = trial_gaps
                    moved = True
            if moved:
                break
        
        temp *= cooling
        if temp < 0.5:
            temp = 0.5
    
    return best

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for the 90s limit
    
    # Baseline: powers of 2 gives length 9
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Phase 1: DFS with backtracking from multiple starting points
    # This is the key improvement: DFS explores more thoroughly than pure greedy
    deadline_dfs = start_time + time_limit * 0.6
    
    # Try DFS from various initial seeds
    initial_seeds = []
    for s in range(1, 25):
        initial_seeds.append([s])
    # Also add some multi-element seeds
    for s in range(1, 15):
        for s2 in range(s + 1, s + 15):
            initial_seeds.append([s, s2])
    
    for seed in initial_seeds:
        if time.time() > deadline_dfs:
            break
        if not _is_valid(seed):
            continue
        result = _dfs_extend(seed, 256, deadline_dfs)
        if len(result) > len(best_seq):
            best_seq = result
    
    # Phase 2: Greedy construction (fast, provides good starting points)
    deadline_greedy = start_time + time_limit * 0.7
    for s in range(1, 20):
        if time.time() > deadline_greedy:
            break
        seq = _greedy_construct(s, 256)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # Phase 3: Local search / simulated annealing on gap vectors
    rng = random.Random(42)
    deadline_ls = start_time + time_limit * 0.9
    
    # Start from the best found
    result = _local_search(best_seq, deadline_ls, rng)
    if len(result) > len(best_seq):
        best_seq = result
    
    # Phase 4: Final targeted extension + local search with remaining time
    deadline_final = start_time + time_limit - 2
    # Try to extend best_seq
    for x in range(best_seq[-1] + 1, 257):
        if time.time() > deadline_final:
            break
        trial = best_seq + [x]
        if _is_valid(trial):
            best_seq = trial
            break
    
    if time.time() < deadline_final:
        result = _local_search(best_seq, deadline_final, rng)
        if len(result) > len(best_seq):
            best_seq = result
    
    # Ensure final validity
    if not _is_valid(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_seq
