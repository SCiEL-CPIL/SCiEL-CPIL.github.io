import time
import random
from math import gcd, isqrt

def construct():
    N = 512
    # Deadline guard
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if lcm(i, j) > N
    # We use Python integers as bitsets
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            # Calculate LCM
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(i, j)
            lcm_val = (i * j) // g
            if lcm_val > N:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Initial set: 1 to isqrt(N)
    # isqrt(512) = 22
    k = isqrt(N)
    S_mask = 0
    for i in range(1, k + 1):
        S_mask |= (1 << i)
    
    best_mask = S_mask
    best_size = k
    
    # Greedy Addition
    # Try to add numbers from k+1 to N
    # Order: Smaller numbers first? Larger numbers have more conflicts.
    # Adding smaller numbers first might block larger ones, but smaller numbers
    # are generally "safer" (fewer conflicts with each other).
    # However, we already have 1..k.
    # Let's try adding in increasing order.
    for x in range(k + 1, N + 1):
        if (S_mask & conflict_mask[x]) == 0:
            S_mask |= (1 << x)
            if S_mask.bit_count() > best_size:
                best_size = S_mask.bit_count()
                best_mask = S_mask
    
    # Local Search
    # We will try random swaps and additions.
    # To make it effective, we should prioritize candidates that are "compatible" with many current members.
    
    # Helper to get list of candidates not in S
    def get_candidates(S_mask):
        return [i for i in range(1, N + 1) if (S_mask & (1 << i)) == 0]
    
    # Helper to get list of members in S
    def get_members(S_mask):
        return [i for i in range(1, N + 1) if (S_mask & (1 << i)) != 0]
    
    # Optimization: Precompute list of all numbers
    all_numbers = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        # Try Additions
        # Pick a random candidate
        # To be faster, we can just iterate or pick random
        # Let's pick random candidates
        candidates = [i for i in all_numbers if (S_mask & (1 << i)) == 0]
        if not candidates:
            break
        
        # Try a few random additions
        for _ in range(10):
            x = random.choice(candidates)
            if (S_mask & conflict_mask[x]) == 0:
                new_mask = S_mask | (1 << x)
                S_mask = new_mask
                if S_mask.bit_count() > best_size:
                    best_size = S_mask.bit_count()
                    best_mask = S_mask
                # Update candidates list is expensive, just re-check or clear if empty
                # We'll just let the loop handle it, or break if no more
                # Actually, if we add x, it might block others.
                # Let's just restart the candidate selection or accept that we might miss some.
                # For efficiency, we'll just continue the loop.
        
        # Try Swaps
        members = [i for i in all_numbers if (S_mask & (1 << i)) != 0]
        if not members:
            break
            
        for _ in range(50):
            y = random.choice(members)
            x = random.choice(candidates)
            if x == y:
                continue
            
            # Check if swapping y with x is valid
            # New set: S - {y} + {x}
            # Condition: x must not conflict with any element in (S - {y})
            # S_mask without y: S_mask ^ (1 << y)
            # Check: (S_mask ^ (1 << y)) & conflict_mask[x] == 0
            if ((S_mask ^ (1 << y)) & conflict_mask[x]) == 0:
                new_mask = (S_mask ^ (1 << y)) | (1 << x)
                # Swapping keeps size same, but might enable future additions
                # We accept the swap if it's valid.
                # To avoid oscillation, we might only accept if it leads to a "better" state later,
                # but for now, we just explore.
                S_mask = new_mask
                # Update best_mask only if size increased, which it didn't here.
                # But we might find a better configuration for size.
                # We track best_size separately.
                
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if (best_mask & (1 << i)) != 0:
            result.append(i)
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
