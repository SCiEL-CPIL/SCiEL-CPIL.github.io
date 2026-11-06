"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 256
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Also need to check a=a: 2a must not be a perfect square
    # Remove self-conflicts from consideration (they're handled by the constraint)
    # Actually, the constraint is: for a<=b in set, a+b is not a perfect square
    # This includes a=a, so 2a must not be a perfect square
    # So numbers where 2a is a perfect square are forbidden entirely
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Use bitmask for speed
    # Map valid numbers to indices 0..M-1
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Greedy initialization: order by degree (ascending), add if no conflict
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: degrees[i])
    
    def greedy_init():
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected
    
    best_mask = greedy_init()
    best_size = bin(best_mask).count('1')
    
    # Local search: try removing one and adding one or two
    def get_size(mask):
        return bin(mask).count('1')
    
    def can_add(mask, i):
        return (adj[i] & mask) == 0
    
    # Iterated local search
    random.seed(42)
    current_mask = best_mask
    current_size = best_size
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Try 1-flip: remove one selected, try to add one or two unselected
        # Or try 2-flip: remove two, add up to three
        
        # Strategy: pick a random selected element to remove, then greedily add
        if current_size == 0:
            break
            
        sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
        if not sel_indices:
            break
        
        # Try removing one random selected element
        remove_idx = random.choice(sel_indices)
        trial_mask = current_mask & ~(1 << remove_idx)
        trial_size = current_size - 1
        
        # Try to add elements greedily
        unselected = [i for i in range(M) if not (trial_mask >> i) & 1]
        random.shuffle(unselected)
        
        added = 0
        for i in unselected:
            if (adj[i] & trial_mask) == 0:
                trial_mask |= (1 << i)
                trial_size += 1
                added += 1
            if added >= 2:  # need net gain of at least 1
                break
        
        if trial_size > current_size:
            current_mask = trial_mask
            current_size = trial_size
            if current_size > best_size:
                best_mask = current_mask
                best_size = current_size
        else:
            # Accept with some probability (simulated annealing-like)
            # Or just reject
            pass
    
    # Also try a different strategy: start fresh with different orderings
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomize the greedy order
        random_order = list(range(M))
        random.shuffle(random_order)
        
        selected = 0
        for i in random_order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        
        sz = bin(selected).count('1')
        if sz > best_size:
            best_mask = selected
            best_size = sz
    
    # Final local search from best
    current_mask = best_mask
    current_size = best_size
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        if current_size == 0:
            break
        
        sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
        if not sel_indices:
            break
        
        remove_idx = random.choice(sel_indices)
        trial_mask = current_mask & ~(1 << remove_idx)
        trial_size = current_size - 1
        
        unselected = [i for i in range(M) if not (trial_mask >> i) & 1]
        random.shuffle(unselected)
        
        added = 0
        for i in unselected:
            if (adj[i] & trial_mask) == 0:
                trial_mask |= (1 << i)
                trial_size += 1
                added += 1
            if added >= 2:
                break
        
        if trial_size > current_size:
            current_mask = trial_mask
            current_size = trial_size
            if current_size > best_size:
                best_mask = current_mask
                best_size = current_size
    
    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
