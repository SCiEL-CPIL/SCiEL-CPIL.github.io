"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
from collections import deque

def _bfs_reset_length(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    """
    N = len(a_trans)
    start_mask = (1 << N) - 1
    
    # Quick check: if a or b is constant, it's not a cycle, but we need to check if it resets
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
        
    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    # Precompute destination masks for each state for a and b
    dest_a = [1 << a_trans[i] for i in range(N)]
    dest_b = [1 << b_trans[i] for i in range(N)]
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Generate next states
        # Apply 'a'
        next_mask_a = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_a |= dest_a[idx]
            m ^= lsb
            
        if next_mask_a not in visited:
            visited.add(next_mask_a)
            queue.append((next_mask_a, dist + 1))
            
        # Apply 'b'
        next_mask_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_b |= dest_b[idx]
            m ^= lsb
            
        if next_mask_b not in visited:
            visited.add(next_mask_b)
            queue.append((next_mask_b, dist + 1))
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        score = _bfs_reset_length(a_trans, b_trans)
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]

    # Strategy 1: Pinwheel Construction
    # The Pinwheel automaton for N states has:
    # a: i -> (i+1) % N for i != N-1, and N-1 -> 0
    # b: i -> i-1 for i != 0, and 0 -> 0 (or other variations)
    # Actually, the standard Pinwheel achieving n^2-n is:
    # a: i -> (i+1) % N
    # b: i -> (i-1) % N for i > 0, and 0 -> 0
    # Let's verify this specific construction.
    
    # Standard Pinwheel:
    # a: 0->1, 1->2, ..., N-1->0
    # b: 0->0, 1->0, 2->1, ..., N-1->N-2
    # This is a known optimal construction.
    
    a_pinwheel = [(i + 1) % N for i in range(N)]
    b_pinwheel = [0] * N
    for i in range(1, N):
        b_pinwheel[i] = i - 1
    try_automaton(a_pinwheel, b_pinwheel)
    
    # Alternative Pinwheel variant:
    # a: 0->1, 1->2, ..., N-1->0
    # b: 0->N-1, 1->0, 2->1, ..., N-1->N-2
    b_pinwheel2 = [N-1] + [0] + list(range(N-2))
    try_automaton(a_pinwheel, b_pinwheel2)
    
    # Another variant:
    # a: 0->1, 1->2, ..., N-1->0
    # b: 0->0, 1->1, 2->0, 3->1, ... (not standard, might not be optimal)
    
    # Strategy 2: Rank-2 Automata (Vorel's construction)
    # For N=2k, rank-2 automata can also achieve high thresholds.
    # However, Pinwheel is generally the canonical example for the upper bound.
    
    # Let's also try a few other systematic structures just in case.
    # "Binary" structure where a is a cycle and b is a specific permutation.
    
    # Since Pinwheel is theoretically optimal (n^2-n), if it synchronizes, it should give 132.
    # If it doesn't (due to specific constraints or my implementation error), we fallback to search.
    
    if best_score >= N * (N - 1):
        return [best_a, best_b]
        
    # Fallback: Local Search if deterministic construction fails or is suboptimal
    import random
    import math
    
    if time.time() < deadline - 5.0:
        random.seed(42)
        
        if best_a is None:
            current_a = [(i + 1) % N for i in range(N)]
            current_b = [0 if i == 0 else i - 1 for i in range(N)]
        else:
            current_a = best_a[:]
            current_b = best_b[:]
        
        current_score = best_score
        best_sa_score = best_score
        best_sa_a = best_a[:] if best_a else current_a[:]
        best_sa_b = best_b[:] if best_b else current_b[:]
        
        # Simple Hill Climbing / Simulated Annealing
        T = 10.0
        cooling_rate = 0.999
        max_iters = 50000
        iter_count = 0
        
        while time.time() < deadline - 2.0 and T > 0.01 and iter_count < max_iters:
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                old_dest = current_a[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_a[i] = new_dest
                new_score = _bfs_reset_length(current_a, current_b)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_sa_score:
                        best_sa_score = new_score
                        best_sa_a = current_a[:]
                        best_sa_b = current_b[:]
                else:
                    delta = new_score - current_score
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_a[i] = old_dest
            else:
                i = random.randint(0, N - 1)
                old_dest = current_b[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_b[i] = new_dest
                new_score = _bfs_reset_length(current_a, current_b)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_sa_score:
                        best_sa_score = new_score
                        best_sa_a = current_a[:]
                        best_sa_b = current_b[:]
                else:
                    delta = new_score - current_score
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_b[i] = old_dest
            
            T *= cooling_rate
            iter_count += 1
        
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_a = best_sa_a
            best_b = best_sa_b
    
    # Final Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
# EVOLVE-BLOCK-END
