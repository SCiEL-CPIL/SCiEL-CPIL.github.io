import time
import random

def _is_valid_fast(seq):
    """Check if seq has all distinct consecutive block sums."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward_optimized():
    selected = []
    sums = set()
    for x in range(1, 257):
        valid = True
        new_sums = []
        current_sum = 0
        for a in reversed(selected):
            current_sum += a
            new_sum = current_sum + x
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.append(x)
        sums.update(new_sums)
    return selected

def _greedy_reverse_optimized():
    selected = []
    sums = set()
    for x in range(256, 0, -1):
        valid = True
        new_sums = []
        current_sum = 0
        for a in selected:
            current_sum += a
            new_sum = x + current_sum
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.insert(0, x)
        sums.update(new_sums)
    return selected

def _get_valid_candidates(values, sums, last_val):
    """Return list of x > last_val such that appending x to values keeps all block sums distinct."""
    valid = []
    n = len(values)
    for x in range(last_val + 1, 257):
        # Check new suffix sums: x, x+values[-1], x+values[-1]+values[-2], ...
        total = x
        if total in sums:
            continue
        ok = True
        for i in range(n - 1, -1, -1):
            total += values[i]
            if total in sums:
                ok = False
                break
        if ok:
            valid.append(x)
    return valid

def _randomized_trial(strategy, rand, time_limit):
    """Run a randomized greedy construction with given strategy."""
    values = []
    sums = set()
    last_val = 0
    
    while last_val < 256:
        if time.time() > time_limit:
            break
        
        candidates = _get_valid_candidates(values, sums, last_val)
        if not candidates:
            break
        
        if strategy == 'random':
            x = rand.choice(candidates)
        elif strategy == 'large':
            k = min(10, len(candidates))
            x = rand.choice(candidates[-k:])
        elif strategy == 'small':
            k = min(10, len(candidates))
            x = rand.choice(candidates[:k])
        elif strategy == 'gap':
            # Prefer candidates that create larger gaps from last_val
            if len(candidates) <= 5:
                x = rand.choice(candidates)
            else:
                # Pick from top half by gap
                gaps = [(c - last_val, c) for c in candidates]
                gaps.sort(reverse=True)
                k = min(10, len(gaps))
                x = rand.choice([g[1] for g in gaps[:k]])
        elif strategy == 'middle':
            # Pick from middle of candidate range
            if len(candidates) <= 5:
                x = rand.choice(candidates)
            else:
                mid_start = len(candidates) // 4
                mid_end = 3 * len(candidates) // 4
                x = rand.choice(candidates[mid_start:mid_end])
        else:
            x = rand.choice(candidates)
        
        values.append(x)
        last_val = x
        
        # Update sums with new suffixes
        total = x
        sums.add(total)
        for a in reversed(values[:-1]):
            total += a
            sums.add(total)
            
    return values

def _local_improve(seq, time_limit):
    """Try to improve sequence by swapping or inserting elements."""
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # Precompute sums for best
    def compute_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            t = 0
            for j in range(i, n):
                t += s[j]
                sums.add(t)
        return sums
    
    best_sums = compute_sums(best)
    n = len(best)
    
    # Try single swaps: swap best[i] with best[j] for i < j, check validity
    # This is expensive, so do limited attempts
    attempts = 0
    max_attempts = 500
    
    for _ in range(max_attempts):
        if time.time() > time_limit:
            break
        attempts += 1
        
        # Try removing one element and reinserting elsewhere
        i = random.randint(0, n - 1)
        val = best[i]
        new_seq = best[:i] + best[i+1:]
        
        # Try inserting val at various positions
        for pos in range(len(new_seq) + 1):
            if pos == i:
                continue
            candidate = new_seq[:pos] + [val] + new_seq[pos:]
            
            # Check strictly increasing
            if any(candidate[k] >= candidate[k+1] for k in range(len(candidate)-1)):
                continue
            
            # Check all values in [1, 256]
            if any(v < 1 or v > 256 for v in candidate):
                continue
            
            # Check distinct consecutive sums
            if _is_valid_fast(candidate):
                if len(candidate) > best_len:
                    best = candidate
                    best_len = len(best)
                    best_sums = compute_sums(best)
                    n = len(best)
                break
    
    # Try inserting a new element not in best
    existing = set(best)
    available = [x for x in range(1, 257) if x not in existing]
    
    for val in available:
        if time.time() > time_limit:
            break
        
        for pos in range(len(best) + 1):
            # Check strictly increasing
            if pos > 0 and best[pos-1] >= val:
                continue
            if pos < len(best) and val >= best[pos]:
                continue
            
            candidate = best[:pos] + [val] + best[pos:]
            
            if _is_valid_fast(candidate):
                best = candidate
                best_len = len(best)
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_seq = []
    
    # 1. Fast deterministic greedies
    seq_fwd = _greedy_forward_optimized()
    if len(seq_fwd) > len(best_seq):
        best_seq = seq_fwd
        
    seq_rev = _greedy_reverse_optimized()
    if len(seq_rev) > len(best_seq):
        best_seq = seq_rev
    
    # 2. Local improvement on best so far
    if time.time() < deadline - 5:
        improved = _local_improve(best_seq, deadline - 5)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    # 3. Randomized trials with multiple strategies
    strategies = ['random', 'large', 'small', 'gap', 'middle', 'random', 'large', 'gap']
    trial_count = 0
    
    while time.time() < deadline - 10:  # Leave 10s for final polish
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count * 7 + 13)
        
        seq = _randomized_trial(strategy, rand, deadline - 10)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            # Try local improvement on new best
            if time.time() < deadline - 5:
                improved = _local_improve(best_seq, deadline - 5)
                if len(improved) > len(best_seq):
                    best_seq = improved
        
        trial_count += 1
        
        # Adaptive: if we've been stuck for a while, try different seed patterns
        if trial_count % 50 == 0 and time.time() < deadline - 15:
            # Try a few more aggressive strategies
            for strat in ['gap', 'large', 'random']:
                if time.time() > deadline - 10:
                    break
                rand2 = random.Random(trial_count * 31 + 7)
                seq2 = _randomized_trial(strat, rand2, deadline - 10)
                if len(seq2) > len(best_seq):
                    best_seq = seq2
    
    # 4. Final local improvement pass with remaining time
    if time.time() < deadline - 2:
        improved = _local_improve(best_seq, deadline - 2)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    return best_seq
