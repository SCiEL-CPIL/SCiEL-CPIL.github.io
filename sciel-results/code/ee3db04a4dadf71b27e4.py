"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _compute_reset_length_bitmask(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    # Precompute transition masks for each state and letter
    # trans_a[state] is the bit representing where state goes under 'a'
    # Actually, we need to map a subset (bitmask) to a new subset.
    # Let's precompute for each state i, the destination bit for a and b.
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    # BFS
    visited = {full_mask}
    # Queue stores (mask, distance)
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
        
        # Transition with 'a'
        new_mask_a = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
            
        # Transition with 'b'
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
            
    return -1

def _is_synchronizing_bitmask(a, b):
    """Check if automaton synchronizes using bitmask BFS."""
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    visited = {full_mask}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        if mask & (mask - 1) == 0:
            return True
        
        new_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append(new_mask_a)
            
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append(new_mask_b)
            
    return False

# EVOLVE-BLOCK-START
def construct():
    # Set deadline to 50s to be safely under the 60s evaluator limit
    deadline = time.time() + 50.0
    
    best_a = None
    best_b = None
    best_len = -1
    
    # Known good initial constructions
    candidates = []
    
    # Construction 1: cycle + decrement
    a1 = list(range(1, N)) + [0]
    b1 = [0] + list(range(N - 1))
    candidates.append((a1, b1))
    
    # Try more diverse initial candidates
    rng_init = random.Random(12345)
    for _ in range(100):
        a = [rng_init.randrange(N) for _ in range(N)]
        b = [rng_init.randrange(N) for _ in range(N)]
        candidates.append((a, b))
    
    # Evaluate all candidates
    for a, b in candidates:
        if time.time() > deadline:
            break
        if _is_synchronizing_bitmask(a, b):
            length = _compute_reset_length_bitmask(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length
    
    # If no synchronizing automaton found, fallback to a known one (should not happen with 100 randoms + 1 known)
    if best_a is None:
        best_a = a1
        best_b = b1
        best_len = _compute_reset_length_bitmask(best_a, best_b)
        if best_len < 0:
             # Should be synchronizing
             return [best_a, best_b]

    # Local search with multiple strategies
    rng = random.Random(42)
    while time.time() < deadline:
        # Perturb the best automaton
        a = list(best_a)
        b = list(best_b)
        
        # Choose perturbation strategy
        strategy = rng.random()
        if strategy < 0.3:
            idx = rng.randrange(N)
            a[idx] = rng.randrange(N)
        elif strategy < 0.6:
            idx = rng.randrange(N)
            b[idx] = rng.randrange(N)
        elif strategy < 0.8:
            idx1 = rng.randrange(N)
            idx2 = rng.randrange(N)
            a[idx1] = rng.randrange(N)
            a[idx2] = rng.randrange(N)
        else:
            idx1 = rng.randrange(N)
            idx2 = rng.randrange(N)
            b[idx1] = rng.randrange(N)
            b[idx2] = rng.randrange(N)
        
        # Quick check for synchronization first? 
        # _is_synchronizing_bitmask is fast.
        if _is_synchronizing_bitmask(a, b):
            length = _compute_reset_length_bitmask(a, b)
            if length > best_len:
                best_a, best_b, best_len = a, b, length
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
