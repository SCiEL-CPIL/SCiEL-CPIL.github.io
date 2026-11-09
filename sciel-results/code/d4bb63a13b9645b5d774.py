"""
Strategy: Simulated Annealing with Bitmask BFS.

The Černý automaton achieves the theoretical bound (N-1)^2 = 121 for N=12.
We verify this immediately. If the bound is reached, we return the optimal solution.
If not (due to any subtle misunderstanding of the metric or bound), we perform a
Simulated Annealing search using a highly optimized bitmask-based BFS to explore
the space of automata, aiming to find a configuration with a higher reset threshold.

The key to performance is the bitmask BFS:
- Subsets of states are represented as 12-bit integers.
- Transitions are precomputed as lookup tables for each letter and each possible subset?
  No, precomputing for all 2^12 subsets is feasible (4096 entries * 2 letters).
  Let's do that. It makes the BFS inner loop extremely fast.
"""

import time
import random
from collections import deque

N = 12
THEORETICAL_MAX = (N - 1) ** 2  # 121

def precompute_transitions(trans):
    """
    Precompute the next state mask for every possible subset mask and each letter.
    trans: [list0, list1]
    Returns: [table0, table1] where tableX is a list of size 2^N.
    tableX[mask] = next_mask after applying letter X to the subset represented by mask.
    """
    size = 1 << N
    table0 = [0] * size
    table1 = [0] * size
    
    # We can compute this iteratively or using DP.
    # For each mask, next_mask = OR of (1 << trans[a][i]) for i in mask.
    # This is O(N * 2^N) which is 12 * 4096 = 49152 ops. Very fast.
    
    for mask in range(size):
        m0 = 0
        m1 = 0
        # Iterate bits
        # Alternatively, we can use a recursive DP:
        # table[mask] = table[mask without lsb] | (1 << trans[lsb])
        
    # Let's use the DP approach for speed.
    table0[0] = 0
    table1[0] = 0
    for mask in range(1, size):
        # Find lowest set bit
        lsb = mask & (-mask)
        prev_mask = mask ^ lsb
        state_idx = lsb.bit_length() - 1
        
        table0[mask] = table0[prev_mask] | (1 << trans[0][state_idx])
        table1[mask] = table1[prev_mask] | (1 << trans[1][state_idx])
        
    return table0, table1

def compute_threshold_precomputed(table0, table1):
    """
    BFS using precomputed transition tables.
    Returns shortest reset word length or inf.
    """
    full_mask = (1 << N) - 1
    
    # visited can be an array of ints for distance, initialized to -1
    size = 1 << N
    visited = [-1] * size
    visited[full_mask] = 0
    
    queue = deque()
    queue.append(full_mask)
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        # Check if single state
        if (current_mask & (current_mask - 1)) == 0:
            return dist
            
        # Next states
        next0 = table0[current_mask]
        next1 = table1[current_mask]
        
        if visited[next0] == -1:
            visited[next0] = dist + 1
            queue.append(next0)
            
        if visited[next1] == -1:
            visited[next1] = dist + 1
            queue.append(next1)
            
    return float('inf')

def generate_pawlik_base():
    """
    Standard Černý automaton.
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return [a_map, b_map]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Check the known optimal solution
    base = generate_pawlik_base()
    table0, table1 = precompute_transitions(base)
    best_score = compute_threshold_precomputed(table0, table1)
    
    if best_score >= THEORETICAL_MAX:
        return base

    # 2. If not at max (unlikely), run Simulated Annealing
    import random
    rng = random.Random(42)
    
    best_trans = base
    best_score_val = best_score
    current_trans = base
    current_score = best_score
    
    # Initialize temperature
    T = 10.0
    cooling_rate = 0.99
    
    # Number of steps per second approx: 
    # Precompute: ~50k ops. BFS: variable, but fast.
    # Let's aim for ~1000 iterations per second.
    
    while time.time() < deadline - 5.0:
        # Mutation: change 1-3 transitions
        candidate = [list(current_trans[0]), list(current_trans[1])]
        num_mutations = rng.randint(1, 3)
        for _ in range(num_mutations):
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            candidate[letter][state] = new_dest
            
        # Evaluate
        c_table0, c_table1 = precompute_transitions(candidate)
        c_score = compute_threshold_precomputed(c_table0, c_table1)
        
        # Acceptance
        if c_score != float('inf'):
            if c_score > current_score:
                current_trans = candidate
                current_score = c_score
                if c_score > best_score_val:
                    best_trans = candidate
                    best_score_val = c_score
                    if best_score_val >= THEORETICAL_MAX:
                        return best_trans
            else:
                # Metropolis criterion
                delta = current_score - c_score
                if T > 0:
                    prob = 1.0 / (1.0 + delta / T)
                    if rng.random() < prob:
                        current_trans = candidate
                        current_score = c_score
        else:
            # If not synchronizing, reject or accept with low prob?
            # Usually reject non-synchronizing in this context unless we want to explore.
            # Let's reject to keep the search focused on synchronizing automata.
            pass
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_trans

# EVOLVE-BLOCK-END
