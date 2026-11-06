"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

def _compute_reset_length_bitmask(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    # Precompute destination bits for each state under a and b
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
    deadline = time.time() + 55.0  # Safety margin for the 60s limit
    best_a = None
    best_b = None
    best_len = -1

    def evaluate(a, b):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return
        # Fast check for synchronizing
        if not _is_synchronizing_bitmask(a, b):
            return
        length = _compute_reset_length_bitmask(a, b)
        if length > best_len:
            best_a, best_b, best_len = list(a), list(b), length

    # Strategy 1: Systematic enumeration of known structural families
    # These are proven to achieve high thresholds, often near-optimal for small N.
    
    # 1a. Pinwheel variants
    # Standard Pinwheel: a is a cycle, b has a fixed point and shifts others
    # a: i -> (i+1) % N
    # b: 0 -> 0, i -> i-1 for i > 0
    for shift_a in range(N):
        a = [(i + 1 + shift_a) % N for i in range(N)]
        b = [0] + list(range(N - 1))
        evaluate(a, b)
        
    # 1b. Rank-2 Cycle Variants
    # a: cycle
    # b: various mappings that create rank-2 subsets
    a_cycle = list(range(1, N)) + [0]
    
    b_variants = [
        [0] + list(range(N - 1)),       # Standard pinwheel
        list(range(N - 1)) + [0],       # Shift left
        [N - 1 - i for i in range(N)],  # Reverse
        [(i + N // 2) % N for i in range(N)], # Half shift
        [0] * N,                        # Trivial
    ]
    
    for b in b_variants:
        evaluate(a_cycle, b)

    # 1c. Systematic search over 'b' for fixed 'a' cycle
    # We know 'a' being a cycle is often optimal.
    # Let's explore 'b' more systematically.
    # b[i] is in 0..N-1.
    # We can't enumerate all 12^12.
    # But we can look at specific structures.
    
    # Structure: b is a permutation? No, usually not.
    # Structure: b has a specific fixed point.
    
    # Let's try to refine the Pinwheel 'b' by changing one or two destinations.
    # This bridges the gap between systematic enumeration and local search.
    
    base_a = list(range(1, N)) + [0]
    base_b = [0] + list(range(N - 1))
    
    # 1-1 perturbations
    for i in range(N):
        for new_dest in range(N):
            if new_dest == base_b[i]:
                continue
            b_copy = base_b[:]
            b_copy[i] = new_dest
            evaluate(base_a, b_copy)
            
    # 2-2 perturbations (only if time permits)
    if time.time() < deadline - 10:
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5:
                    break
                # Limit search space for 2-perturbations
                # Only try a few destinations for each
                for new_dest_i in [0, (base_b[i] + 1) % N, (base_b[i] - 1) % N]:
                    for new_dest_j in [0, (base_b[j] + 1) % N, (base_b[j] - 1) % N]:
                        if new_dest_i == base_b[i] and new_dest_j == base_b[j]:
                            continue
                        b_copy = base_b[:]
                        b_copy[i] = new_dest_i
                        b_copy[j] = new_dest_j
                        evaluate(base_a, b_copy)

    # Strategy 2: Local Search (Hill Climbing) starting from the best found so far
    # This refines the structural candidates.
    
    if best_a is None:
        best_a = base_a[:]
        best_b = base_b[:]
        best_len = _compute_reset_length_bitmask(best_a, best_b)
        if best_len <= 0:
            # Fallback
            return [base_a, base_b]

    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_len
    
    rng = random.Random(42)
    
    # Simulated Annealing / Hill Climbing hybrid
    # We accept worse moves occasionally to escape local optima.
    # But since we have a strict time limit and want the *exact* max,
    # strict hill climbing from a good start is often safer.
    # Let's do strict hill climbing first, then random restarts if stuck.
    
    steps = 0
    max_steps = 10000 # Safety cap, time is the real limit
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        improved = False
        
        # Try a batch of mutations
        for _ in range(50):
            if time.time() > deadline:
                break
                
            # Pick random state and letter
            state = rng.randint(0, N - 1)
            letter = rng.randint(0, 1)
            
            if letter == 0:
                old_val = current_a[state]
                new_val = rng.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_a[state] = new_val
                # Check synchronizing first to save time on reset length calc?
                # _compute_reset_length_bitmask returns -1 if not sync.
                new_score = _compute_reset_length_bitmask(current_a, current_b)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_len:
                        best_len = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    # Revert
                    current_a[state] = old_val
            else:
                old_val = current_b[state]
                new_val = rng.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_b[state] = new_val
                new_score = _compute_reset_length_bitmask(current_a, current_b)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_len:
                        best_len = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    # Revert
                    current_b[state] = old_val
                    
        # If stuck for a while, restart from global best or a random structural candidate
        if not improved and steps % 100 == 0:
            # Restart from global best
            current_a = best_a[:]
            current_b = best_b[:]
            current_score = best_len
            
            # Or try a random structural candidate
            if rng.random() < 0.3:
                # Random Pinwheel variant
                shift = rng.randint(0, N - 1)
                current_a = [(i + 1 + shift) % N for i in range(N)]
                current_b = [0] + list(range(N - 1))
                current_score = _compute_reset_length_bitmask(current_a, current_b)
                if current_score > best_len:
                    best_len = current_score
                    best_a = current_a[:]
                    best_b = current_b[:]

    return [best_a, best_b]
# EVOLVE-BLOCK-END
