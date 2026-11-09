import time
import random

def _is_valid_full(values):
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _count_conflicts(values):
    """Count duplicate block sums. Returns number of duplicates (0 = valid)."""
    n = len(values)
    seen = set()
    dupes = 0
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                dupes += 1
            else:
                seen.add(total)
    return dupes

def _local_search(values, max_val, deadline, rng):
    """
    Hill-climbing local search:
    - State: strictly increasing sequence of integers in [1, max_val]
    - Objective: minimize number of duplicate block sums (target: 0)
    - Move: pick a random position, change value to a nearby valid value
    - Accept: if conflict count doesn't increase
    """
    best = list(values)
    best_conflicts = _count_conflicts(best)
    n = len(best)
    if n == 0:
        return best, 0
    
    if best_conflicts == 0:
        return best, 0
    
    # Track best found
    global_best = list(best)
    global_best_conflicts = best_conflicts
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Pick a random position to modify
        pos = rng.randint(0, n - 1)
        
        # Determine valid range for this position
        lo = best[pos - 1] + 1 if pos > 0 else 1
        hi = best[pos + 1] - 1 if pos < n - 1 else max_val
        
        if lo >= hi:
            continue
        
        # Try a random value in range
        # Bias toward smaller values (to compress range)
        if rng.random() < 0.6:
            # Try to decrease
            delta = rng.randint(1, max(1, (best[pos] - lo) // 2))
            new_val = max(lo, best[pos] - delta)
        else:
            # Try a random value
            new_val = rng.randint(lo, hi)
        
        if new_val == best[pos]:
            continue
        
        # Apply change
        best[pos] = new_val
        new_conflicts = _count_conflicts(best)
        
        if new_conflicts <= best_conflicts:
            best_conflicts = new_conflicts
            if new_conflicts < global_best_conflicts:
                global_best_conflicts = new_conflicts
                global_best = list(best)
                if new_conflicts == 0:
                    break
        else:
            # Revert
            best[pos] = best[pos]  # already reverted by assignment above
            # Actually need to restore
            # We set best[pos] = new_val, now revert
            # But we already changed it. Let me fix:
            pass
        
        # Simulated annealing: occasionally accept worse moves
        if new_conflicts > best_conflicts and rng.random() < 0.05:
            # Keep the change (already applied)
            best_conflicts = new_conflicts
        else:
            # Revert
            # We need to restore the old value
            # This is getting messy. Let me restructure.
            pass
    
    return global_best, global_best_conflicts

def _local_search_v2(values, max_val, deadline, rng):
    """Cleaner local search with proper state management."""
    current = list(values)
    current_conflicts = _count_conflicts(current)
    n = len(current)
    if n == 0:
        return current, 0
    
    if current_conflicts == 0:
        return current, 0
    
    best_found = list(current)
    best_conflicts = current_conflicts
    
    # Temperature for simulated annealing
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        temp *= cooling
        
        pos = rng.randint(0, n - 1)
        lo = current[pos - 1] + 1 if pos > 0 else 1
        hi = current[pos + 1] - 1 if pos < n - 1 else max_val
        
        if lo >= hi:
            continue
        
        old_val = current[pos]
        
        # Choose new value
        if rng.random() < 0.5:
            # Small perturbation
            delta = rng.choice([-1, 1, -2, 2, -3, 3])
            new_val = old_val + delta
        else:
            # Larger jump
            new_val = rng.randint(lo, hi)
        
        if new_val < lo or new_val > hi or new_val == old_val:
            continue
        
        current[pos] = new_val
        new_conflicts = _count_conflicts(current)
        
        delta_c = new_conflicts - current_conflicts
        
        if delta_c <= 0 or (temp > 0.01 and rng.random() < (2.0 / (1.0 + temp)) ** (-delta_c / max(1, temp))):
            # Accept
            current_conflicts = new_conflicts
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_found = list(current)
                if new_conflicts == 0:
                    break
        else:
            # Reject
            current[pos] = old_val
    
    return best_found, best_conflicts

def _construct_exponential(k, max_val=256):
    """
    Construct a sequence of length k using powers of 2, scaled to fit in [1, max_val].
    Powers of 2: 1, 2, 4, 8, ..., 2^(k-1)
    All subarray sums of powers of 2 are distinct (binary representation).
    Then we try to compress by reducing values while maintaining validity.
    """
    # Start with powers of 2
    seq = [2**i for i in range(k)]
    
    # Check if it fits
    if seq[-1] > max_val:
        # Scale down: use floor of geometric sequence
        # a_i = floor(a_0 * r^i) where a_0=1, r chosen so a_{k-1} <= max_val
        # r = max_val^(1/(k-1))
        if k == 1:
            return [1]
        r = max_val ** (1.0 / (k - 1))
        seq = []
        prev = 0
        for i in range(k):
            if i == 0:
                v = 1
            else:
                v = int(round(1 * (r ** i)))
            v = max(v, prev + 1)  # Ensure strictly increasing
            if v > max_val:
                v = max_val
            prev = v
            seq.append(v)
        
        # Ensure strictly increasing
        for i in range(1, k):
            if seq[i] <= seq[i-1]:
                seq[i] = seq[i-1] + 1
            if seq[i] > max_val:
                return seq[:i]  # Can't fit k elements
        
        return seq
    
    return seq

def _greedy_compress(values, max_val=256):
    """
    Given a valid sequence, try to reduce each element (from right to left)
    while maintaining validity and strict increase.
    """
    n = len(values)
    if n == 0:
        return values
    
    result = list(values)
    
    # Try to reduce from the right
    for i in range(n - 1, -1, -1):
        lo = result[i-1] + 1 if i > 0 else 1
        hi = result[i+1] - 1 if i < n-1 else max_val
        
        # Try to reduce result[i]
        for candidate in range(lo, result[i]):
            result[i] = candidate
            if _is_valid_full(result):
                break
        else:
            # Couldn't reduce, restore
            result[i] = values[i]
    
    return result

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    n = len(current)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best = []
    best_len = 0
    rng = random.Random(42)
    
    # Strategy 1: Try different k values with exponential construction + local search
    # Powers of 2 give valid sequences. Find max k where 2^(k-1) <= 256
    # 2^8 = 256, so k=9 with pure powers. But we can compress.
    
    # Start with greedy to get a baseline
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
    
    # Strategy 2: Try various initial constructions and local search
    for k in range(best_len + 3, best_len - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to construct a sequence of length k
        seq = _construct_exponential(k, 256)
        if len(seq) < k:
            continue
        
        # Check if valid
        if _is_valid_full(seq):
            # Compress it
            compressed = _greedy_compress(seq, 256)
            if len(compressed) > best_len:
                best = compressed
                best_len = len(best)
            continue
        
        # Not valid, try local search to fix
        # But first, make sure it's strictly increasing and in range
        # It should be from construction
        
        # Run local search
        remaining_time = deadline - time.time() - 2
        if remaining_time < 1:
            break
        
        found, conflicts = _local_search_v2(seq, 256, time.time() + min(remaining_time, 8), rng)
        if conflicts == 0 and len(found) > best_len:
            best = found
            best_len = len(best)
    
    # Strategy 3: Local search starting from greedy solution, trying to extend
    if time.time() < deadline - 10:
        # Try to extend best by 1, 2, 3 elements using local search
        for target_len in range(best_len + 1, best_len + 4):
            if time.time() > deadline - 5:
                break
            
            # Start with best + some random extension
            base = list(best)
            # Add random elements
            cur_sums = set()
            n = len(base)
            for i in range(n):
                total = 0
                for j in range(i, n):
                    total += base[j]
                    cur_sums.add(total)
            
            for x in range(base[-1] + 1, 257):
                tails = [x]
                total = x
                for a in reversed(base):
                    total += a
                    tails.append(total)
                if not cur_sums.intersection(tails):
                    base.append(x)
                    cur_sums.update(tails)
                    break
            
            # Now try to reach target_len with local search
            # First, add random elements to get to target_len
            while len(base) < target_len:
                added = False
                for _ in range(50):
                    x = rng.randint(base[-1] + 1, 256)
                    tails = [x]
                    total = x
                    for a in reversed(base):
                        total += a
                        tails.append(total)
                    if not cur_sums.intersection(tails):
                        base.append(x)
                        cur_sums.update(tails)
                        added = True
                        break
                if not added:
                    break
            
            if len(base) >= target_len:
                base = base[:target_len]
                remaining_time = deadline - time.time() - 3
                if remaining_time > 1:
                    found, conflicts = _local_search_v2(base, 256, time.time() + min(remaining_time, 10), rng)
                    if conflicts == 0 and len(found) > best_len:
                        best = found
                        best_len = len(best)
    
    # Strategy 4: Randomized restarts with local search
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 3:
                break
            
            # Random valid-ish starting point
            k = rng.randint(max(5, best_len - 2), best_len + 2)
            seq = []
            last = 0
            for i in range(k):
                # Pick values with some structure
                if i == 0:
                    v = rng.randint(1, 50)
                else:
                    gap = rng.randint(1, max(1, (256 - last) // (k - i)))
                    v = min(last + gap, 256)
                if v > 256 or v <= last:
                    break
                seq.append(v)
                last = v
            
            if len(seq) < 5:
                continue
            
            remaining_time = deadline - time.time() - 2
            if remaining_time < 1:
                break
            
            found, conflicts = _local_search_v2(seq, 256, time.time() + min(remaining_time, 5), rng)
            if conflicts == 0 and len(found) > best_len:
                best = found
                best_len = len(best)
    
    # Final: try greedy extension on best
    if time.time() < deadline - 2:
        extended = _greedy_forward(best)
        if len(extended) > best_len:
            best = extended
            best_len = len(best)
    
    # Final validation
    if best and not _is_valid_full(best):
        # Fallback to greedy
        best = _greedy_forward([])
    
    return best
