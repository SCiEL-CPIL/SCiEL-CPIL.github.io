import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses DP-based O(2^N) construction for subset transitions.
    Returns -1 if not synchronizing.
    """
    # Precompute transition masks for each state and letter
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # DP-based construction of next subsets: O(2^N)
    # next_subsets[s][0] is the image of subset s under letter 0
    # next_subsets[s][1] is the image of subset s under letter 1
    next_subsets = [[0, 0] for _ in range(1 << N)]
    
    # Base case: empty set maps to empty set
    # next_subsets[0][0] = 0, next_subsets[0][1] = 0 (already initialized)
    
    # For each non-empty subset, pick the least significant bit
    # s = lsb | rest
    # next(s) = next(rest) | t[lsb_idx]
    for s in range(1, 1 << N):
        lsb = s & -s
        idx = (lsb.bit_length() - 1)
        rest = s ^ lsb
        
        # We can compute this recursively or iteratively.
        # Since we are iterating s from 1 to 2^N-1, rest < s is guaranteed.
        # So next_subsets[rest] is already computed.
        next_subsets[s][0] = next_subsets[rest][0] | t0[idx]
        next_subsets[s][1] = next_subsets[rest][1] | t1[idx]
        
    # BFS
    visited = [-1] * (1 << N)
    # Use a list-based queue for speed if possible, or deque.
    # Deque is fine, but list with pointer is faster in Python sometimes.
    # Let's stick to deque for clarity unless we need max speed.
    # Actually, for BFS on 4096 nodes, overhead is minimal.
    
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current & (current - 1) == 0 and current != 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current][letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_pinwheel_seed(rng):
    """
    Generates a pinwheel-like automaton.
    Pinwheel automata are known to have reset threshold N-1 or higher.
    Structure:
    a: 0->1, 1->2, ..., N-2->N-1, N-1->0 (Cycle)
    b: 0->0, 1->0, ..., N-1->N-2 (Shift down, 0 fixed)
    This is the standard Černy automaton.
    
    To get "pinwheel" variations, we can modify the cycle or the shift.
    A common pinwheel structure involves a cycle where one state points to itself or breaks the cycle slightly,
    and a second letter that shifts the rest.
    
    Let's generate a random near-permutation for b that acts as a "defective" shift.
    """
    trans0 = [(i + 1) % N for i in range(N)]
    
    # trans1: Start with a shift, then introduce 1-2 defects to ensure synchronization
    # If trans1 is a pure shift (i -> i-1), and trans0 is a cycle, it's the standard Cerny.
    # To vary it, we can change the target of 0 or 1 in trans1.
    
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
        
    # Introduce a defect: map one state to a non-neighbor
    # This breaks the perfect pinwheel but often maintains high threshold
    defect_idx = rng.randint(1, N-1)
    # Map it to 0 or N-1 or random
    # Mapping to 0 is common
    trans1[defect_idx] = rng.choice([0, N-1, rng.randint(0, N-1)])
    
    return trans0, trans1

def _generate_near_permutation_seed(rng):
    """
    Generates a random near-permutation automaton.
    One letter is a random permutation.
    The other is a random function with rank N-1 or N-2.
    """
    trans0 = list(range(N))
    rng.shuffle(trans0)
    
    trans1 = list(range(N))
    # Create a defect: merge two states
    i = rng.randint(0, N-1)
    j = rng.randint(0, N-1)
    while j == i:
        j = rng.randint(0, N-1)
    trans1[j] = trans1[i]
    
    # Shuffle trans1 slightly to break structure? No, keep it close to permutation.
    # Actually, just shuffling a near-permutation is good.
    # But we want it to be "near" a shift or cycle for pinwheel properties.
    # Let's just return this.
    
    return trans0, trans1

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety (Limit is 90s)
    
    rng = random.Random(42)
    
    best_0_global = None
    best_1_global = None
    best_score_global = 0
    
    # Beam Search
    # Initialize with Pinwheel seeds and Near-Permutation seeds
    beam = []
    
    # Add standard Cerny (Pinwheel base)
    c0, c1 = _generate_pinwheel_seed(rng)
    score = _compute_threshold_fast(c0, c1)
    if score > 0:
        beam.append((c0[:], c1[:], score))
        
    # Add variations
    for _ in range(20):
        t0, t1 = _generate_pinwheel_seed(rng)
        score = _compute_threshold_fast(t0, t1)
        if score > 0:
            beam.append((t0[:], t1[:], score))
            
    for _ in range(20):
        t0, t1 = _generate_near_permutation_seed(rng)
        score = _compute_threshold_fast(t0, t1)
        if score > 0:
            beam.append((t0[:], t1[:], score))
            
    # Sort and keep top K
    beam.sort(key=lambda x: x[2], reverse=True)
    K = 50
    beam = beam[:K]
    
    # Update global best
    if beam:
        best_score_global = beam[0][2]
        best_0_global = beam[0][0][:]
        best_1_global = beam[0][1][:]
        
    seen = set()
    for t0, t1, s in beam:
        seen.add((tuple(t0), tuple(t1)))
        
    while time.time() < deadline:
        if not beam:
            break
            
        # Expand the top candidate (or a random top candidate to diversify)
        # Let's expand the top one to focus search
        cur_0, cur_1, cur_score = beam[0]
        
        # Generate neighbors by mutating one transition
        neighbors = []
        
        # Mutate letter 0
        for s in range(N):
            old_val = cur_0[s]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                mut_0 = cur_0[:]
                mut_0[s] = new_val
                mut_1 = cur_1[:]
                
                # Quick check: if both are permutations, skip?
                # If both are permutations, they form a group, no sync.
                # Check rank
                if len(set(mut_0)) == N and len(set(mut_1)) == N:
                    continue
                    
                new_score = _compute_threshold_fast(mut_0, mut_1)
                if new_score > 0:
                    key = (tuple(mut_0), tuple(mut_1))
                    if key not in seen:
                        seen.add(key)
                        neighbors.append((mut_0, mut_1, new_score))
                        
        # Mutate letter 1
        for s in range(N):
            old_val = cur_1[s]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                mut_0 = cur_0[:]
                mut_1 = cur_1[:]
                mut_1[s] = new_val
                
                if len(set(mut_0)) == N and len(set(mut_1)) == N:
                    continue
                    
                new_score = _compute_threshold_fast(mut_0, mut_1)
                if new_score > 0:
                    key = (tuple(mut_0), tuple(mut_1))
                    if key not in seen:
                        seen.add(key)
                        neighbors.append((mut_0, mut_1, new_score))
                        
        # Merge neighbors into beam
        beam.extend(neighbors)
        
        # Sort and keep top K
        beam.sort(key=lambda x: x[2], reverse=True)
        beam = beam[:K]
        
        # Update global best
        if beam[0][2] > best_score_global:
            best_score_global = beam[0][2]
            best_0_global = beam[0][0][:]
            best_1_global = beam[0][1][:]
            
    # Fallback
    if best_0_global is None:
        b0, b1 = _generate_pinwheel_seed(rng)
        return [b0, b1]
        
    return [best_0_global, best_1_global]

# EVOLVE-BLOCK-END
