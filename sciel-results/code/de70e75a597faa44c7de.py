"""
Structural change: fast backtracking with randomized restarts,
iterative deepening on target length, and local repair.
Uses set-based sum tracking with O(1) membership tests.
Randomized candidate ordering provides diversity across restarts.
"""
import time
import random
import sys
from collections import defaultdict

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _construct_greedy_randomized(seed):
    """Greedy with randomized value ordering."""
    rng = random.Random(seed)
    values, sums = [], set()
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    for x in candidates:
        if values and x <= values[-1]:
            continue
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _backtrack_fast(deadline, target_len=None, rng=None):
    """
    Fast backtracking using a set of block sums.
    Randomized candidate ordering for diversity across restarts.
    """
    if rng is None:
        rng = random.Random()
    
    best_seq = []
    best_len = 0
    
    def _try_extend(seq, sums, last_val):
        """Yield (v, new_sums_list) for each valid extension value v."""
        n = len(seq)
        for v in range(last_val + 1, 257):
            total = v
            new_sums_list = [total]
            valid = True
            for i in range(n - 1, -1, -1):
                total += seq[i]
                if total in sums:
                    valid = False
                    break
                new_sums_list.append(total)
            if valid:
                yield v, new_sums_list
    
    def dfs(seq, sums, last_val, depth):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        if target_len is not None and n >= target_len:
            return
        
        # Pruning
        if n + (256 - last_val) <= best_len:
            return
        if target_len is not None and n + (256 - last_val) < target_len:
            return
        
        # Generate candidates
        candidates = list(_try_extend(seq, sums, last_val))
        if not candidates:
            return
        
        # Randomize order
        rng.shuffle(candidates)
        
        # Limit branching
        max_branch = 20
        if len(candidates) > max_branch:
            candidates.sort(key=lambda x: x[0])
            half = len(candidates) // 2
            chosen = candidates[:half]
            if half < max_branch:
                remaining = candidates[half:]
                if remaining:
                    chosen.extend(rng.sample(remaining, min(len(remaining), max_branch - half)))
            candidates = chosen[:max_branch]
        
        for v, new_sums_list in candidates:
            new_seq = seq + [v]
            new_sums = sums.union(new_sums_list)
            dfs(new_seq, new_sums, v, depth + 1)
            if time.time() > deadline:
                return
            if target_len is not None and best_len >= target_len:
                return
    
    dfs([], set(), 0, 0)
    return best_seq

def _local_repair(seq, deadline, rng):
    """Try to extend or repair a sequence."""
    if not _check_valid(seq):
        return seq
    
    best_seq = list(seq)
    
    # Try extending
    for _ in range(50):
        if time.time() > deadline:
            break
        last = best_seq[-1] if best_seq else 0
        candidates = []
        sums = set()
        n = len(best_seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_seq[j]
                sums.add(s)
        for v in range(last + 1, 257):
            total = v
            ok = True
            for i in range(n - 1, -1, -1):
                total += best_seq[i]
                if total in sums:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        
        if not candidates:
            break
        
        v = rng.choice(candidates)
        new_seq = best_seq + [v]
        if _check_valid(new_seq):
            best_seq = new_seq
    
    # Try swapping
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(best_seq) < 2:
            break
        i = rng.randint(0, len(best_seq) - 1)
        old_val = best_seq[i]
        min_val = best_seq[i-1] + 1 if i > 0 else 1
        max_val = best_seq[i+1] - 1 if i < len(best_seq) - 1 else 256
        if min_val >= max_val:
            continue
        new_val = rng.randint(min_val, max_val)
        if new_val == old_val:
            continue
        test_seq = best_seq[:i] + [new_val] + best_seq[i+1:]
        if _check_valid(test_seq):
            best_seq = test_seq
    
    return best_seq

def _multi_restart(deadline):
    """Multiple randomized restarts with iterative deepening."""
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    # Phase 1: greedy variants
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        seq = _construct_greedy_randomized(seed)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 2: backtracking with increasing targets
    target = best_len + 1
    max_target = best_len + 8
    while target <= max_target and time.time() < deadline - 5:
        found = False
        for attempt in range(4):
            if time.time() > deadline - 5:
                break
            rng = random.Random(target * 7919 + attempt)
            seq = _backtrack_fast(deadline - 5, target_len=target, rng=rng)
            if len(seq) >= target:
                best_seq = seq
                best_len = len(seq)
                found = True
                break
        if not found:
            break
        target = best_len + 1
    
    # Phase 3: local repair
    if time.time() < deadline - 3:
        rng = random.Random(12345)
        repaired = _local_repair(best_seq, deadline - 3, rng)
        if len(repaired) > best_len:
            best_seq = repaired
            best_len = len(repaired)
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    try:
        result = _multi_restart(deadline)
        if len(result) > best_len:
            best = result
            best_len = len(best)
    except Exception:
        pass
    
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
