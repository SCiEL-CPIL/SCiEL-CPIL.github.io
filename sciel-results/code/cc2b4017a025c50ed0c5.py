import time
import random
from collections import defaultdict
import numpy as np

def _compute_counts_numpy(chosen):
    """Compute sum representation counts using numpy boolean array."""
    n = len(chosen)
    if n == 0:
        return np.zeros(2001, dtype=np.int32)
    
    # Create boolean indicator array
    arr = np.zeros(1001, dtype=np.int32)
    for x in chosen:
        arr[x] = 1
    
    # Convolution gives number of ways to write s as a+b where a,b in [1,1000]
    # But we need a <= b, so we compute full convolution and adjust
    conv = np.convolve(arr, arr, mode='full')
    
    # conv[s] = number of ordered pairs (a,b) with a+b=s, a,b in chosen
    # We need unordered pairs with a <= b
    # For s even: count = (conv[s] + 1) // 2 if s/2 is in chosen, else conv[s] // 2
    # Actually: unordered pairs with a<=b:
    #   If s is even and s/2 in chosen: (conv[s] - 1) // 2 + 1 = (conv[s]+1)//2
    #   Otherwise: conv[s] // 2
    
    counts = np.zeros(2001, dtype=np.int32)
    for s in range(2, 2001):
        if s % 2 == 0 and s // 2 in range(1, 1001) and arr[s // 2] == 1:
            counts[s] = (conv[s] + 1) // 2
        else:
            counts[s] = conv[s] // 2
    
    return counts

def _check_valid_numpy(chosen_arr, counts):
    """Check if all counts are <= 2."""
    return np.all(counts <= 2)

def _try_greedy_fast(order, use_numpy=True):
    """Fast greedy using numpy for collision checking."""
    chosen = []
    chosen_set = set()
    counts = np.zeros(2001, dtype=np.int32)
    
    for x in order:
        if x in chosen_set:
            continue
        
        # Check if adding x is valid
        # New sums: x + y for y in chosen, and 2*x
        # We need counts[s] + 1 <= 2 for all new sums s
        
        valid = True
        
        # Check sums with existing elements
        if chosen:
            # Vectorized check
            new_sums = np.array([x + y for y in chosen])
            # Check if any new sum would exceed 2
            if np.any(counts[new_sums] + 1 > 2):
                valid = False
        
        if valid:
            s2 = 2 * x
            if counts[s2] + 1 > 2:
                valid = False
        
        if valid:
            # Add x
            if chosen:
                new_sums = np.array([x + y for y in chosen])
                counts[new_sums] += 1
            counts[2 * x] += 1
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _dense_random_init(rng, target_size=700):
    """Create a dense random initialization, then prune violations."""
    # Start with a random subset of target_size elements
    candidates = list(range(1, 1001))
    rng.shuffle(candidates)
    initial = candidates[:target_size]
    
    # Compute counts and find violations
    chosen = sorted(initial)
    counts = _compute_counts_numpy(chosen)
    
    # Iteratively remove elements that cause violations
    # Priority: remove elements involved in the most violations
    max_iters = 1000
    for _ in range(max_iters):
        # Find sums with count > 2
        bad_sums = np.where(counts > 2)[0]
        if len(bad_sums) == 0:
            break
        
        # For each bad sum, find pairs that contribute to it
        # We'll track which elements are "blamed" most
        blame = defaultdict(int)
        
        for s in bad_sums:
            s_int = int(s)
            # Find all pairs (a,b) in chosen with a+b=s, a<=b
            for i, a in enumerate(chosen):
                b = s_int - a
                if b < a:
                    continue
                if b in chosen:
                    # This pair contributes to the violation
                    blame[a] += 1
                    blame[b] += 1
        
        if not blame:
            break
        
        # Remove the most blamed element
        worst_elem = max(blame.keys(), key=lambda x: blame[x])
        if worst_elem in chosen:
            chosen.remove(worst_elem)
            # Recompute counts
            counts = _compute_counts_numpy(chosen)
        else:
            break
    
    return chosen

def _local_search_numpy(chosen, time_limit=50.0):
    """
    Local search using numpy for fast collision checking.
    1. Try to insert any valid element.
    2. Try to swap: remove one, add another.
    """
    chosen = sorted(chosen)
    n = len(chosen)
    if n == 0:
        return chosen
    
    # Initialize counts using numpy
    counts = _compute_counts_numpy(chosen)
    chosen_set = set(chosen)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Try Insertions
        candidates = [x for x in range(1, 1001) if x not in chosen_set]
        if not candidates:
            break
        rng = random.Random(int(time.time() * 1000) % 10000)
        rng.shuffle(candidates)
        
        for y in candidates[:100]:  # Try up to 100 candidates per pass
            if time.time() - start_time > time_limit:
                break
            
            # Check if y can be added
            valid = True
            if chosen:
                new_sums = np.array([y + x for x in chosen])
                if np.any(counts[new_sums] + 1 > 2):
                    valid = False
            
            if valid:
                s2 = 2 * y
                if counts[s2] + 1 > 2:
                    valid = False
            
            if valid:
                # Add y
                if chosen:
                    new_sums = np.array([y + x for x in chosen])
                    counts[new_sums] += 1
                counts[2 * y] += 1
                chosen.append(y)
                chosen_set.add(y)
                n += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try Swaps
        # Try to swap one element out and one in
        indices_to_try = list(range(n))
        rng = random.Random(int(time.time() * 1000) % 10000)
        rng.shuffle(indices_to_try)
        
        for idx in indices_to_try[:50]:  # Try up to 50 removals per pass
            if time.time() - start_time > time_limit:
                break
            
            x = chosen[idx]
            
            # Temporarily remove x
            # Update counts: we need to recompute since numpy approach is easier
            chosen_without_x = chosen[:idx] + chosen[idx+1:]
            counts_without_x = _compute_counts_numpy(chosen_without_x)
            
            # Try to add a new element y
            swap_found = False
            # Try random candidates, biased towards values that fill gaps
            for _ in range(30):
                y = random.randint(1, 1000)
                if y in chosen_set and y != x:
                    continue
                if y == x:
                    continue
                    
                valid = True
                if chosen_without_x:
                    new_sums = np.array([y + z for z in chosen_without_x])
                    if np.any(counts_without_x[new_sums] + 1 > 2):
                        valid = False
                
                if valid:
                    s2 = 2 * y
                    if counts_without_x[s2] + 1 > 2:
                        valid = False
                
                if valid:
                    # Add y
                    if chosen_without_x:
                        new_sums = np.array([y + z for z in chosen_without_x])
                        counts_without_x[new_sums] += 1
                    counts_without_x[2 * y] += 1
                    chosen = chosen_without_x + [y]
                    chosen.sort()
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    counts = counts_without_x
                    swap_found = True
                    improved = True
                    break
            
            if not swap_found:
                # Revert: we haven't changed chosen, so nothing to revert
                pass
    
    return chosen

def _polish(chosen):
    """Try to add any remaining valid elements."""
    chosen = sorted(chosen)
    counts = _compute_counts_numpy(chosen)
    chosen_set = set(chosen)
    
    for y in range(1, 1001):
        if y in chosen_set:
            continue
        
        valid = True
        if chosen:
            new_sums = np.array([y + x for x in chosen])
            if np.any(counts[new_sums] + 1 > 2):
                valid = False
        
        if valid:
            s2 = 2 * y
            if counts[s2] + 1 > 2:
                valid = False
        
        if valid:
            if chosen:
                new_sums = np.array([y + x for x in chosen])
                counts[new_sums] += 1
            counts[2 * y] += 1
            chosen.append(y)
            chosen_set.add(y)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, 1001)),
        list(range(1000, 0, -1)),
    ]
    
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(1, 1001))
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        s = _try_greedy_fast(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Strategy 2: Dense random initialization + pruning
    if time.time() < deadline - 15.0:
        for seed in range(5):
            if time.time() > deadline - 15.0:
                break
            rng = random.Random(seed + 200)
            s = _dense_random_init(rng, target_size=700)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)
    
    # Strategy 3: Local search on the best found so far
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 5.0
        improved = _local_search_numpy(best_set, time_limit=time_left)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Strategy 4: More restarts with local search
    if time.time() < deadline - 10.0:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            rng = random.Random(seed + 300)
            order = list(range(1, 1001))
            rng.shuffle(order)
            s = _try_greedy_fast(order)
            
            time_left = deadline - time.time() - 5.0
            if time_left > 2.0:
                s = _local_search_numpy(s, time_limit=min(5.0, time_left * 0.5))
            
            if len(s) > best_size:
                best_set = s
                best_size = len(s)
    
    # Final Polish
    if time.time() < deadline - 1.0:
        polished = _polish(best_set)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
    
    return best_set

# Test/Run
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
