import random
import time

N = 256

def is_sidon(S):
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    # Compute blocked set B using optimized approach
    # y is blocked if exists a in S such that y+a in sums, OR 2y in sums
    blocked = bytearray(N + 1)  # blocked[y] = 1 if y is blocked
    
    # For each sum s, mark all y such that y + a = s for some a in S
    # i.e., y = s - a for some a in S
    # Also mark y = s/2 if s is even
    
    for s in sums:
        for a in S:
            y = s - a
            if 1 <= y <= N:
                blocked[y] = 1
        if s % 2 == 0:
            y = s // 2
            if 1 <= y <= N:
                blocked[y] = 1
    
    # Check if all y not in S are blocked
    for y in range(1, N + 1):
        if y in S_set:
            continue
        if not blocked[y]:
            return False
    return True

def greedy_sidon(order, N):
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def try_remove_one(S, N, time_limit=None):
    """Try to remove one element from S while maintaining maximality.
    Returns (new_S, removed) if successful, (S, None) otherwise."""
    S_set = set(S)
    
    # Precompute sums
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    # For each candidate removal, check if remaining set is maximal
    # To speed up, precompute which elements would become unblocked
    # if we remove x
    
    # Strategy: try removing each element in S, check maximality
    # Optimize by checking in random order and returning first success
    
    candidates = list(S)
    random.shuffle(candidates)
    
    for x in candidates:
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Check if S \ {x} is still maximal
        # S \ {x} is maximal iff every y not in S\{x} is blocked by S\{x}
        # y is blocked by S\{x} iff:
        #   - y in S\{x} (already in set)
        #   - exists a in S\{x} such that y+a in sums(S\{x})
        #   - 2y in sums(S\{x})
        
        # sums(S\{x}) = sums(S) minus sums involving x
        # This is expensive to compute, so just call is_maximal directly
        # but with a quick pre-check: if removing x unblocks some y, fail
        
        # Quick pre-check: find if any y was ONLY blocked by x
        # y is only blocked by x if:
        #   - y not in S
        #   - for all a in S\{x}, y+a not in sums(S) (or y+a was a sum involving x)
        #   - 2y not in sums(S) (or 2y was a sum involving x)
        # This is complex, so just do the full check
        
        candidate_S = [s for s in S if s != x]
        if is_maximal(candidate_S, N):
            return candidate_S, x
    
    return S, None

def try_swap(S, N, time_limit=None):
    """Try to swap one element out and one in while maintaining Sidon + maximality."""
    S_set = set(S)
    
    # Precompute sums
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    # For each x in S, try swapping with some y not in S
    # To be efficient, only try y that are "close" to being valid
    
    for x in S:
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Try a few random y values
        for _ in range(10):
            y = random.randint(1, N)
            if y in S_set:
                continue
            
            # Check if S - {x} + {y} is Sidon
            # New sums involving y: y+a for a in S\{x}, and 2y
            # Old sums removed: x+a for a in S\{x}, and 2x
            # Need: all new sums distinct from each other and from remaining old sums
            
            remaining = [s for s in S if s != x]
            
            # Check new sums
            new_sums = set()
            valid = True
            for a in remaining:
                s = y + a
                if s in new_sums or s in sums:
                    # s in sums means it conflicts with an existing sum
                    # but we need to be careful: sums includes sums involving x
                    # Actually, we need sums of remaining only
                    valid = False
                    break
                new_sums.add(s)
            
            if not valid:
                continue
            
            s2 = 2 * y
            if s2 in new_sums or s2 in sums:
                continue
            
            # Also need to check that new_sums don't conflict with remaining sums
            # remaining sums = sums - {x+a for a in S} - {2x}
            # This is getting complex, just do full check
            
            candidate_S = remaining + [y]
            if not is_sidon(candidate_S):
                continue
            if is_maximal(candidate_S, N):
                return candidate_S
    
    return S

def reduce_maximal(S, N, time_limit=None):
    S = list(S)
    improved = True
    while improved:
        improved = False
        
        # Try removals
        new_S, removed = try_remove_one(S, N, time_limit)
        if removed is not None:
            S = new_S
            improved = True
            continue
        
        # Try swaps
        new_S = try_swap(S, N, time_limit)
        if len(new_S) == len(S) and set(new_S) != set(S):
            S = new_S
            improved = True
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 25  # Safety margin
    
    best = None
    best_size = N + 1
    
    random.seed(42)
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = reduce_maximal(S, N, deadline)
    if len(S) < best_size:
        best = S
        best_size = len(S)
    
    # Try multiple random orderings with different strategies
    for trial in range(10000):
        if time.time() > deadline:
            break
        
        # Different randomization strategies
        strategy = trial % 4
        
        if strategy == 0:
            # Pure random
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy == 1:
            # Random with bias toward smaller numbers
            order = list(range(1, N + 1))
            for i in range(len(order)):
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif strategy == 2:
            # Start with some fixed small numbers, then random
            base = [random.randint(1, 20) for _ in range(5)]
            base = list(set(base))
            rest = [x for x in range(1, N + 1) if x not in base]
            random.shuffle(rest)
            order = base + rest
        else:
            # Reverse random
            order = list(range(N, 0, -1))
            random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce with time limit
        remaining_time = deadline - time.time()
        if remaining_time > 1:
            S = reduce_maximal(S, N, deadline - 0.5)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
            # If we found a small set, do more focused search
            if best_size <= 20:
                # Try more aggressive local search on best
                for _ in range(50):
                    if time.time() > deadline - 1:
                        break
                    # Random perturbation
                    perturbed = list(best)
                    # Remove a random element
                    if len(perturbed) > 1:
                        idx = random.randint(0, len(perturbed) - 1)
                        perturbed.pop(idx)
                    # Add a random element
                    candidates = [x for x in range(1, N + 1) if x not in perturbed]
                    if candidates:
                        new_elem = random.choice(candidates)
                        perturbed.append(new_elem)
                    
                    if is_sidon(perturbed):
                        perturbed = reduce_maximal(perturbed, N, deadline - 0.5)
                        if len(perturbed) < best_size:
                            best = perturbed
                            best_size = len(perturbed)
    
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
