"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_valid(seq):
    """Check if all nonempty consecutive block sums are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(start_values, candidates, existing_sums=None):
    """Greedy build: add values from candidates in order, checking validity."""
    values = list(start_values)
    if not values:
        values = []
    
    # Compute existing sums if we have a starting sequence
    if values:
        if existing_sums is None:
            sums = set()
            n = len(values)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += values[j]
                    sums.add(s)
            existing_sums = sums
        sums = existing_sums
    else:
        sums = set()
    
    for x in candidates:
        if values and x <= values[-1]:
            continue
        # Check if adding x creates duplicate consecutive block sums
        # New sums are: x, values[-1]+x, values[-2]+values[-1]+x, ..., sum(values)+x
        # i.e., suffix sums of values + x
        tails = []
        total = 0
        for a in reversed(values):
            total += a
            tails.append(total)
        tails.append(x)  # the single element x
        
        # Actually let me recompute: new consecutive blocks ending at x
        # are: [x], [values[-1], x], [values[-2], values[-1], x], ..., [values[0], ..., values[-1], x]
        # Their sums are: x, values[-1]+x, values[-2]+values[-1]+x, ...
        
        conflict = False
        total = 0
        for a in reversed(values):
            total += a
            s = total + x
            if s in sums:
                conflict = True
                break
        if not conflict:
            if x in sums:
                conflict = True
        
        if not conflict:
            values.append(x)
            # Update sums
            total = 0
            for a in reversed(values[:-1]):  # values without the last x
                total += a
                sums.add(total + x)
            sums.add(x)
    
    return values

def _try_build_with_gaps(min_val, max_val, n_target, time_limit):
    """Try to build a sequence of length n_target in [min_val, max_val] using greedy with random restarts."""
    deadline = time.time() + time_limit
    best = []
    
    # Strategy: pick n_target values from [min_val, max_val] and try to order them
    # Since they must be strictly increasing, we just need to pick which values to use.
    # The greedy approach: try to pick values that maximize the chance of valid consecutive block sums.
    
    # Key insight: larger gaps between consecutive elements tend to make block sums more distinct.
    # So we want to spread values out.
    
    # Approach: parameterized by step size
    # Try different step sizes and offsets
    
    step_min = (max_val - min_val + 1) // n_target
    if step_min < 1:
        step_min = 1
    
    # Try various strategies
    strategies = []
    
    # Strategy 1: Evenly spaced
    if n_target <= max_val - min_val + 1:
        step = (max_val - min_val + 1) // n_target
        for offset in range(min(5, step)):
            candidates = [min_val + offset + i * step for i in range(n_target)]
            if candidates[-1] <= max_val:
                strategies.append(candidates)
    
    # Strategy 2: Start from high end
    for offset in range(min(5, step_min)):
        candidates = [max_val - offset - i * step_min for i in range(n_target)]
        candidates.reverse()
        if candidates[0] >= min_val and candidates[-1] <= max_val:
            strategies.append(candidates)
    
    # Strategy 3: Random subsets
    num_random = 20
    for _ in range(num_random):
        # Pick n_target random values from [min_val, max_val]
        if max_val - min_val + 1 >= n_target:
            candidates = sorted(random.sample(range(min_val, max_val + 1), n_target))
            strategies.append(candidates)
    
    # Strategy 4: Exponential-like spacing
    for base_mult in [1.2, 1.3, 1.5, 2.0]:
        candidates = []
        v = min_val
        while v <= max_val and len(candidates) < n_target:
            candidates.append(v)
            v = int(v * base_mult) + 1
        if len(candidates) == n_target and candidates[-1] <= max_val:
            strategies.append(candidates)
    
    # Evaluate all strategies
    for candidates in strategies:
        if time.time() > deadline:
            break
        if len(candidates) < n_target:
            continue
        seq = _greedy_build([], candidates, set())
        if len(seq) > len(best):
            best = seq
        if len(best) >= n_target:
            break
    
    return best

def _dfs_build(values, candidates, sums, n_target, time_limit, deadline, best_seq):
    """DFS to build a valid sequence, trying to reach n_target length."""
    if len(values) == n_target:
        if len(values) > len(best_seq):
            best_seq.extend(values)
        return True
    
    if time.time() > deadline:
        return False
    
    for i, x in enumerate(candidates):
        if values and x <= values[-1]:
            continue
        
        # Check if adding x is valid
        conflict = False
        total = 0
        for a in reversed(values):
            total += a
            s = total + x
            if s in sums:
                conflict = True
                break
        if not conflict and x in sums:
            conflict = True
        
        if not conflict:
            # Add x
            values.append(x)
            new_sums = set()
            total = 0
            for a in reversed(values[:-1]):
                total += a
                new_sums.add(total + x)
            new_sums.add(x)
            sums.update(new_sums)
            
            remaining = candidates[i+1:]
            if _dfs_build(values, remaining, sums, n_target, time_limit, deadline, best_seq):
                values.pop()
                sums.difference_update(new_sums)
                return True
            
            values.pop()
            sums.difference_update(new_sums)
    
    return False

def construct():
    """Return a strictly increasing sequence of integers in [1,256] with distinct consecutive block sums.
    Maximize sequence length."""
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_seq = []
    
    # First, get a baseline using the original greedy approach
    values, sums = [], set()
    for x in range(1, 257):
        tails = []
        total = 0
        for a in reversed(values):
            total += a
            tails.append(total)
        tails.append(x)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best_seq = values
    best_len = len(best_seq)
    
    if time.time() > deadline:
        return best_seq
    
    # Now try to improve with various strategies
    # Target: try to get longer sequences
    
    # Try DFS with smaller candidate sets for larger n
    for n_target in range(best_len + 1, 20):
        if time.time() > deadline - 5:
            break
        
        # Try to build a sequence of length n_target
        # Use a reduced candidate set to make DFS feasible
        # Pick evenly spaced candidates
        num_candidates = min(n_target * 3, 40)
        
        # Multiple attempts with different candidate sets
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Evenly spaced
                step = 256 // num_candidates
                candidates = [i * step + 1 for i in range(num_candidates)]
                candidates = [c for c in candidates if 1 <= c <= 256]
            else:
                # Random
                candidates = sorted(random.sample(range(1, 257), min(num_candidates, 256)))
            
            if len(candidates) < n_target:
                continue
            
            # Try DFS
            values = []
            sums = set()
            best_local = []
            
            def dfs_helper(vals, cands, sm, bt):
                if len(vals) == n_target:
                    if len(vals) > len(bt):
                        bt.extend(vals)
                    return True
                if time.time() > deadline - 3:
                    return False
                for i, x in enumerate(cands):
                    if vals and x <= vals[-1]:
                        continue
                    conflict = False
                    total = 0
                    for a in reversed(vals):
                        total += a
                        s = total + x
                        if s in sm:
                            conflict = True
                            break
                    if not conflict and x in sm:
                        conflict = True
                    if not conflict:
                        vals.append(x)
                        new_sums = set()
                        total = 0
                        for a in reversed(vals[:-1]):
                            total += a
                            new_sums.add(total + x)
                        new_sums.add(x)
                        sm.update(new_sums)
                        
                        if dfs_helper(vals, cands[i+1:], sm, bt):
                            vals.pop()
                            sm.difference_update(new_sums)
                            return True
                        vals.pop()
                        sm.difference_update(new_sums)
                return False
            
            dfs_helper(values, candidates, sums, best_local)
            
            if len(best_local) > best_len:
                best_len = len(best_local)
                best_seq = best_local
    
    # Try greedy with larger gaps
    for gap in range(1, 10):
        if time.time() > deadline - 2:
            break
        candidates = list(range(1, 257, gap))
        seq = _greedy_build([], candidates, set())
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    # Try starting from high values
    for start in range(256, 1, -10):
        if time.time() > deadline - 2:
            break
        candidates = list(range(1, start + 1, max(1, (start - 1) // 30)))
        seq = _greedy_build([], candidates, set())
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    return best_seq

# EVOLVE-BLOCK-END
