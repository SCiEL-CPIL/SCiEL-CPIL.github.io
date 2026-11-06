import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Optimized for speed using precomputed transition tables for all subsets.
    Returns -1 if not synchronizing.
    """
    # Precompute single-state transitions as bitmasks
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Precompute next subset for every possible subset and both letters
    # next_subsets[s, 0] is the subset reached from s by letter 0
    # next_subsets[s, 1] is the subset reached from s by letter 1
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    for s in range(1 << N):
        res0 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res0 |= t0[idx]
            temp_s ^= lsb
        next_subsets[s, 0] = res0
        
        res1 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res1 |= t1[idx]
            temp_s ^= lsb
        next_subsets[s, 1] = res1

    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if current subset is a single state (synchronized)
        if current != 0 and (current & (current - 1)) == 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_ping_pong_defect(cycle0, cycle1, defect_state, defect_dest0, defect_dest1):
    """
    Generates a ping-pong automaton with a specific defect.
    cycle0: list representing the permutation for letter 0 (usually a cycle)
    cycle1: list representing the permutation for letter 1 (usually a cycle)
    defect_state: the state where the pattern is broken
    defect_dest0: destination for defect_state under letter 0
    defect_dest1: destination for defect_state under letter 1
    """
    trans0 = cycle0[:]
    trans1 = cycle1[:]
    
    # Apply defect
    trans0[defect_state] = defect_dest0
    trans1[defect_state] = defect_dest1
    
    return trans0, trans1

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = 0
    best_0 = None
    best_1 = None
    
    # Standard cycles for N=12
    # Cycle 0: 0->1->2...->11->0
    cycle0 = [(i + 1) % N for i in range(N)]
    # Cycle 1: 0->11->10...->1->0 (Reverse cycle)
    cycle1 = [(i - 1) % N for i in range(N)]
    
    # Strategy: Systematically explore the space of "Ping-Pong" automata.
    # A ping-pong automaton consists of two cycles. Synchronization is achieved
    # by breaking the symmetry with a "defect".
    # The reset word length is roughly 2*N or 3*N depending on the defect.
    
    # We will iterate over all possible defects:
    # 1. Choose the state to break (N options)
    # 2. Choose the new destination for that state under letter 0 (N options)
    # 3. Choose the new destination for that state under letter 1 (N options)
    # Total: 12 * 12 * 12 = 1728 candidates.
    # Each candidate takes < 1ms to evaluate with the optimized BFS.
    # Total time ~ 1-2 seconds. This leaves ample time for more complex structures.
    
    candidates = []
    
    # 1. Basic Ping-Pong with single defect
    for d_state in range(N):
        for d_dest0 in range(N):
            for d_dest1 in range(N):
                t0, t1 = _generate_ping_pong_defect(cycle0, cycle1, d_state, d_dest0, d_dest1)
                candidates.append((t0, t1))
                
    # 2. Variant: Cycle 0 is forward, Cycle 1 is forward (same direction, different start?)
    # Actually, if both are forward cycles, they are just relabelings.
    # Let's try Cycle 0: forward, Cycle 1: forward but shifted? No, that's isomorphic.
    
    # 3. Variant: Cycle 0 is forward, Cycle 1 is a "jump" cycle?
    # Let's stick to the most promising: Opposing cycles (Ping-Pong).
    
    # 4. Variant: Two defects
    # This increases search space significantly. 12^4 = 20736. Still feasible.
    # But let's first evaluate the single defect ones to see the baseline.
    
    # Sort candidates? No, just evaluate all.
    
    # To maximize throughput, we can batch evaluate or just loop.
    # 1728 * 1ms = 1.7s.
    
    # Let's add a second layer: "Double Defect" Ping-Pong
    # Break two states.
    # To keep it manageable, we'll only do this if time permits.
    
    # First pass: Single Defect
    for t0, t1 in candidates:
        if time.time() > deadline:
            break
        score = _compute_threshold_fast(t0, t1)
        if score > best_score:
            best_score = score
            best_0 = t0
            best_1 = t1

    # If we have time, explore "Double Defect" structures
    # We will sample a subset of double defects to avoid combinatorial explosion
    # or just do it all if fast enough.
    # 12^4 = 20736. 20736 * 1ms = 20s.
    # We have ~80s left. So we can do it.
    
    double_defect_candidates = []
    # To avoid duplicates and redundancy, we can just iterate.
    # But 20k evaluations might take 20-30s.
    # Let's optimize: only check if the single defect version was "close" to synchronizing?
    # No, double defects can be much better.
    
    # Let's generate double defect candidates systematically but with a cap.
    # We'll pick the top 100 single-defect structures and perturb them?
    # No, let's just do a full search of double defects if time allows.
    
    # Actually, let's be smarter.
    # The best ping-pong automata often have the defect creating a "bottleneck" or "funnel".
    
    # Let's run the double defect search.
    # To speed it up, we can skip if the first defect makes it non-synchronizing?
    # No, the second defect might fix it.
    
    # We will iterate over d_state1, d_dest0_1, d_dest1_1 and d_state2, d_dest0_2, d_dest1_2.
    # To keep it efficient, we'll limit the second defect to be different from the first.
    
    # Given the time constraint, we'll do a limited double defect search.
    # We'll sample 5000 random double defects from the space.
    
    rng = random.Random(42)
    
    # Pre-compute base cycles
    base0 = cycle0[:]
    base1 = cycle1[:]
    
    # Sample 5000 double defects
    for _ in range(5000):
        if time.time() > deadline - 5: # Leave 5s for final check
            break
            
        # Pick two distinct states
        s1, s2 = rng.sample(range(N), 2)
        
        # Pick random destinations
        d0_1 = rng.randint(0, N-1)
        d1_1 = rng.randint(0, N-1)
        d0_2 = rng.randint(0, N-1)
        d1_2 = rng.randint(0, N-1)
        
        t0 = base0[:]
        t1 = base1[:]
        
        t0[s1] = d0_1
        t1[s1] = d1_1
        t0[s2] = d0_2
        t1[s2] = d1_2
        
        score = _compute_threshold_fast(t0, t1)
        if score > best_score:
            best_score = score
            best_0 = t0
            best_1 = t1
            
    # If best is still 0, return a known synchronizing one (Cerny)
    if best_score == 0 or best_0 is None:
        # Cerny automaton
        best_0 = [(i + 1) % N for i in range(N)]
        best_1 = [0] * N
        for i in range(1, N):
            best_1[i] = i - 1
        best_score = _compute_threshold_fast(best_0, best_1)
        
    return [best_0, best_1]

# EVOLVE-BLOCK-END
