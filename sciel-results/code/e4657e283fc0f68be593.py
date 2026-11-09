import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute conflicts
    # conflicts[x] is a list of bitmasks (integers) representing the set of 3 other elements
    # that would form a 4-AP with x.
    conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x is 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            a, b, c = x + d, x + 2*d, x + 3*d
            if c <= N:
                mask_conf = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                conflicts[x].append(mask_conf)
        
        # x is 2nd term: x-d, x, x+d, x+2d
        max_d = min((x - 1), (N - x) // 2)
        for d in range(1, max_d + 1):
            a, b, c = x - d, x + d, x + 2*d
            if a >= 1 and c <= N:
                mask_conf = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                conflicts[x].append(mask_conf)
                
        # x is 3rd term: x-2d, x-d, x, x+d
        max_d = min((x - 2) // 2, N - x)
        for d in range(1, max_d + 1):
            a, b, c = x - 2*d, x - d, x + d
            if a >= 1 and c <= N:
                mask_conf = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                conflicts[x].append(mask_conf)
                
        # x is 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a, b, c = x - 3*d, x - 2*d, x - d
            if a >= 1:
                mask_conf = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                conflicts[x].append(mask_conf)

    def can_add(mask, x):
        # Check if adding x to mask creates a 4-AP
        for c in conflicts[x]:
            if (mask & c) == c:
                return False
        return True

    def get_size(mask):
        return bin(mask).count("1")

    # Initial greedy construction
    mask = 0
    for x in range(1, N + 1):
        if can_add(mask, x):
            mask |= (1 << (x-1))
    
    best_mask = mask
    best_size = get_size(mask)
    
    # Local Search
    # We will try to improve by adding elements.
    # If stuck, we perform a "kick" (remove random elements) and try again.
    
    while time.time() < deadline:
        # Try to add any element
        improved = False
        for x in range(1, N + 1):
            if not (mask & (1 << (x-1))):
                if can_add(mask, x):
                    mask |= (1 << (x-1))
                    improved = True
                    if get_size(mask) > best_size:
                        best_size = get_size(mask)
                        best_mask = mask
        
        if not improved:
            # Stuck in local optimum. Perform a random restart/kick.
            # Remove a few random elements to allow new ones.
            # Number of elements to remove: start small, increase if failing.
            # Let's just do a single random removal and then greedy fill.
            # Or better: Simulated Annealing style.
            
            # Pick a random subset of current elements to remove
            # Let's try removing 1 random element first.
            if best_size < 1: break # Safety
            
            # Get list of indices in mask
            indices = [i+1 for i in range(N) if (mask >> i) & 1]
            if not indices:
                break
            
            # Try a few random kicks
            kicked = False
            for _ in range(10): # Try 10 random kicks per stall
                if time.time() > deadline: break
                
                # Remove 1 random element
                rem_idx = random.choice(indices)
                new_mask = mask & ~(1 << (rem_idx-1))
                
                # Try to fill new_mask greedily
                for x in range(1, N + 1):
                    if not (new_mask & (1 << (x-1))):
                        if can_add(new_mask, x):
                            new_mask |= (1 << (x-1))
                
                new_size = get_size(new_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                    mask = new_mask
                    kicked = True
                    break
                elif new_size == best_size:
                    # Accept equal size to explore? 
                    # Let's accept to escape plateau
                    mask = new_mask
                    kicked = True
                    break
                    
            if not kicked:
                # If we couldn't even find an equal size path, try removing 2?
                # Or just restart completely?
                # Let's try removing 2 random elements
                if len(indices) >= 2:
                    rem1, rem2 = random.sample(indices, 2)
                    new_mask = mask & ~(1 << (rem1-1)) & ~(1 << (rem2-1))
                    for x in range(1, N + 1):
                        if not (new_mask & (1 << (x-1))):
                            if can_add(new_mask, x):
                                new_mask |= (1 << (x-1))
                    new_size = get_size(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                    elif new_size == best_size:
                        mask = new_mask

    # Convert best_mask to list
    result = [i+1 for i in range(N) if (best_mask >> i) & 1]
    return result
