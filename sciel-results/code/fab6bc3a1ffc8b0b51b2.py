import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflict mask for all pairs
    # mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # Helper to check compatibility with current mask
    def is_compatible(x, current_mask):
        return (conflict_mask[x] & current_mask) == 0
        
    def count_bits(mask):
        c = 0
        while mask:
            c += 1
            mask &= mask - 1
        return c
            
    # Conflict counts for each number
    conflict_counts = [0] * (n + 1)
    for i in range(1, n + 1):
        conflict_counts[i] = count_bits(conflict_mask[i])
    
    # Sort numbers by conflict count (fewer conflicts = safer to add)
    order_by_conflict = list(range(1, n + 1))
    order_by_conflict.sort(key=lambda x: conflict_counts[x])
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy from empty, sorted by conflict count
    current_mask = 0
    for x in order_by_conflict:
        if is_compatible(x, current_mask):
            current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask
    
    # Strategy 2: Greedy from primes+1 start (original)
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    initial_mask = 0
    for i in range(1, n + 1):
        if is_prime[i] or i == 1:
            initial_mask |= (1 << i)
            
    current_mask = initial_mask
    for x in order_by_conflict:
        if not (current_mask >> x) & 1:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask
    
    # Multiple randomized greedy restarts from empty set
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 0.5))  # ~0.5s per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation of order: shuffle with some bias toward low conflict
        order = list(range(1, n + 1))
        # Partial shuffle: keep low-conflict items somewhat earlier
        rng.shuffle(order)
        # Sort by (conflict_count + random_noise) to introduce diversity
        order.sort(key=lambda x: conflict_counts[x] + rng.random() * 5)
        
        current_mask = 0
        for x in order:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
        sz = count_bits(current_mask)
        if sz > best_size:
            best_size = sz
            best_mask = current_mask
    
    # Local search: add-only pass
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if is_compatible(x, best_mask):
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True
    
    # 1-swap local search: try removing y and adding x
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                # x is not in set; check if we can add it by removing one element
                # Find all elements in best_mask that conflict with x
                conflicts_with_x = conflict_mask[x] & best_mask
                if conflicts_with_x == 0:
                    # Can add directly
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
                # Try removing one conflicting element y and adding x
                # Only beneficial if x conflicts with exactly one element y in the set
                # (net zero change in size, but might enable future additions)
                # For now, only do swaps that maintain or increase size
                # A swap removes y and adds x: size stays same
                # But it might allow adding more elements later
                # Only do swap if x has fewer total conflicts than y
                c = conflicts_with_x
                # Get the list of conflicting elements
                conflicting = []
                while c:
                    lb = c & (-c)
                    y = lb.bit_length() - 1
                    conflicting.append(y)
                    c ^= lb
                
                if len(conflicting) == 1:
                    y = conflicting[0]
                    # Check: after removing y and adding x, is the result valid?
                    # x is compatible with (best_mask without y) by construction
                    # We just need to verify x doesn't conflict with itself (trivially true)
                    new_mask = (best_mask & ~(1 << y)) | (1 << x)
                    # This is valid if x is compatible with new_mask minus x
                    # which is best_mask minus y
                    # We already know conflict_mask[x] & (best_mask & ~(1 << y)) == 0
                    # So this swap is valid
                    # Do the swap only if it might help: x has fewer conflicts than y
                    if conflict_counts[x] < conflict_counts[y]:
                        best_mask = new_mask
                        improved = True
                        break
    
    # Final add-only pass
    if time.time() < deadline - 0.5:
        for x in range(1, n + 1):
            if not (best_mask >> x) & 1:
                if is_compatible(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
    
    # Convert back to list
    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return best
