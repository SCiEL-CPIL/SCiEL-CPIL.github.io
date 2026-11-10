import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    # 1. Precompute 4-APs and constraints
    # constraints[x] will be a list of masks (int) representing the other 3 elements
    # in each 4-AP containing x.
    # Bit i is set if element (i+1) is in the mask.
    constraints = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Create masks for each element
            # For a: others are b, c, e
            mask_bce = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            constraints[a].append(mask_bce)
            
            # For b: others are a, c, e
            mask_ace = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            constraints[b].append(mask_ace)
            
            # For c: others are a, b, e
            mask_abe = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            constraints[c].append(mask_abe)
            
            # For e: others are a, b, c
            mask_abc = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            constraints[e].append(mask_abc)

    def can_add(x, S_mask):
        """Check if x can be added to set represented by S_mask"""
        for m in constraints[x]:
            if (S_mask & m) == m:
                return False
        return True

    def greedy_fill(S_mask, order):
        """Greedily add elements from order to S_mask if possible"""
        S = S_mask
        for x in order:
            # Check if x is already in S
            if (S >> (x - 1)) & 1:
                continue
            if can_add(x, S):
                S |= (1 << (x - 1))
        return S

    # 2. Initial Candidates
    candidates = []
    
    # Base-4 set (Salem-Spencer style)
    # Numbers in [1, 512] with base-4 digits in {0, 1}
    base4_mask = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_mask |= (1 << (i - 1))
    
    # Start from base4 and fill
    candidates.append(greedy_fill(base4_mask, range(1, N + 1)))
    candidates.append(greedy_fill(base4_mask, range(N, 0, -1)))
    
    # Greedy from scratch with various orders
    candidates.append(greedy_fill(0, range(1, N + 1)))
    candidates.append(greedy_fill(0, range(N, 0, -1)))
    
    # Randomized greedy
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidates.append(greedy_fill(0, order))
        
    # Select best initial candidate
    best_mask = max(candidates, key=lambda m: bin(m).count('1'))
    best_size = bin(best_mask).count('1')
    
    # 3. Local Search
    # We will perform 1-exchange and 2-exchange moves.
    # A move consists of removing a subset R from S, and then greedily refilling.
    
    def try_remove_and_refill(S_mask, remove_set):
        """Remove elements in remove_set from S_mask, then greedily refill."""
        S = S_mask
        for x in remove_set:
            S &= ~(1 << (x - 1))
        
        # Refill: iterate through all possible elements
        # To make it faster, we can iterate in a fixed order or random order.
        # Fixed order (1 to N) is deterministic and fast.
        # Random order might escape local minima better but is slower if we restart.
        # Let's use a mixed strategy: iterate 1..N.
        for x in range(1, N + 1):
            if (S >> (x - 1)) & 1:
                continue
            if can_add(x, S):
                S |= (1 << (x - 1))
        return S

    # Simulated Annealing / Random Restart Local Search
    current_mask = best_mask
    current_size = best_size
    
    # Precompute list of elements for random selection
    all_elements = list(range(1, N + 1))
    
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        
        # Determine move type
        # 80% 1-exchange, 20% 2-exchange
        if random.random() < 0.8:
            # 1-exchange
            if current_size == 0: break
            # Pick a random element to remove
            # Efficient way: pick random index in 0..current_size-1, find k-th set bit?
            # Simpler: pick random x in 1..N, if in set, use it. Else retry.
            # Or just pick from list of set bits.
            # Since N=512, checking 512 bits is fast.
            elems_in_set = [i for i in range(1, N + 1) if (current_mask >> (i - 1)) & 1]
            if not elems_in_set:
                break
            remove_elem = random.choice(elems_in_set)
            remove_set = [remove_elem]
        else:
            # 2-exchange
            if current_size < 2: break
            elems_in_set = [i for i in range(1, N + 1) if (current_mask >> (i - 1)) & 1]
            if len(elems_in_set) < 2:
                break
            remove_set = random.sample(elems_in_set, 2)
            
        # Perform move
        new_mask = try_remove_and_refill(current_mask, remove_set)
        new_size = bin(new_mask).count('1')
        
        # Acceptance criterion
        if new_size >= current_size:
            current_mask = new_mask
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
        else:
            # Simulated Annealing: accept with probability
            # T decreases over time
            # Simple version: accept if random() < exp(-(current_size - new_size) / 5.0)
            # Since we only care about max, we can just reject.
            # But SA helps escape.
            delta = current_size - new_size
            if delta <= 3: # Small penalty
                if random.random() < 0.3: # 30% chance to accept small loss
                    current_mask = new_mask
                    current_size = new_size
            # If large loss, reject

    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
            
    return result
