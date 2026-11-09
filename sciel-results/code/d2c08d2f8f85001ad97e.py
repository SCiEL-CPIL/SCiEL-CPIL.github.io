"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # The problem: Find a subset S of {1, ..., 256} such that:
    # 1. Elements are used in increasing order (implicit if we pick subset and sort).
    #    Actually, the task says "strictly increasing sequence of integers in [1, 256]".
    #    This implies the sequence is a subset of [1, 256] sorted in ascending order.
    # 2. All non-empty consecutive block sums are distinct.
    # 3. Maximize the length of the sequence.
    
    # Let x_i be 1 if integer i is in the sequence, 0 otherwise.
    # The sequence is sorted, so the consecutive blocks are determined by the positions of 1s in x.
    # However, ILP variables are binary. The "consecutive block" definition depends on the *relative*
    # positions in the selected subsequence, not the original indices.
    # This makes a direct ILP formulation complex because the "consecutive" relation is non-linear
    # with respect to the selection variables x_i.
    
    # Alternative Approach:
    # Since the sequence must be increasing, it is a subset.
    # Let the selected values be v_1 < v_2 < ... < v_k.
    # The sums are S_{i,j} = sum(v_i ... v_j).
    # We need S_{i,j} != S_{i',j'} for all distinct pairs.
    
    # This is a combinatorial search problem. ILP is hard to set up directly for "consecutive"
    # because "consecutive" refers to the indices in the *result* sequence, which depend on x.
    
    # Let's rethink. Is there a simpler structural property?
    # The seed program is greedy. It picks 1, then 2, etc., checking sums.
    # Score 1.0 means it found a sequence of length L_seed.
    # The inspirations found ~1.2x improvement.
    
    # Let's try a different search strategy: **Simulated Annealing** or **Genetic Algorithm**?
    # Or **Local Search**?
    
    # Actually, let's look at the constraints more closely.
    # If we pick a subset, the number of consecutive blocks is k*(k+1)/2.
    # For k=10, 55 sums. For k=15, 120 sums.
    # Max possible sum is 256*15 ~ 3840.
    # Distinct sums constraint is the bottleneck.
    
    # Let's try a **Backtracking Search with Pruning** but optimized with **Bitmasks** or **Bitsets** for sum tracking?
    # The seed uses a set of sums.
    
    # A different design family: **Constraint Satisfaction Problem (CSP) solved via Python's `python-constraint` or custom backtracking with strong heuristics**.
    # The inspirations mentioned DFS, Beam, and Constraint Propagation.
    
    # Let's try **Iterative Deepening Depth-First Search (IDDFS)** with a strong heuristic ordering.
    # Or better: **A* Search** or **Greedy with Backtracking (Limited)**.
    
    # Wait, the inspirations already did DFS, Beam, and Constraint Propagation.
    # I need something *structurally different*.
    
    # How about **Randomized Local Search (Hill Climbing / Simulated Annealing)**?
    # State: A set of indices (or values) from 1..256.
    # Objective: Maximize length, subject to valid distinct sums.
    # Moves: Add a new element, remove an element, swap an element.
    # This explores the space of subsets differently than sequential construction.
    
    # Let's implement a **Simulated Annealing** approach.
    # State: A sorted list of selected values.
    # Validity: Check if all consecutive block sums are distinct.
    # Cost: -length (we want to maximize length). Or simply, we want to find a valid state with max length.
    # Actually, SA is good for optimizing a continuous function. Here we have a hard constraint (validity) and a soft objective (length).
    # We can define an energy function: E = -length + penalty * (number_of_conflicts).
    # If we can find a state with 0 conflicts and high length, that's good.
    
    # However, checking validity is O(k^3) or O(k^2 log k) if done smartly.
    # k is likely small (< 20). So validity check is fast.
    
    # Let's try **Greedy Construction with Randomized Restart and Backtracking**.
    # But that's similar to the seed.
    
    # Let's try **Constructive Heuristic with Local Repair**.
    # 1. Start with a random subset of size L.
    # 2. Check conflicts.
    # 3. If conflicts, remove one element from a conflicting pair.
    # 4. Try to add a new element.
    
    # Actually, a very different approach: **Metropolis-Hastings (MCMC)** to sample valid sequences and track the longest one.
    
    # Let's go with **Simulated Annealing** on the set of selected values.
    # State: frozenset of values.
    # Neighborhood: Add one value, remove one value.
    # Energy: -len(state) * 1000 + conflict_count * 1.
    # We want to minimize Energy.
    
    def get_conflicts(seq):
        """Return number of duplicate consecutive block sums."""
        if not seq:
            return 0
        sums = set()
        conflicts = 0
        # seq is sorted
        # Compute all consecutive sums
        # O(k^2) sums
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in sums:
                    conflicts += 1
                else:
                    sums.add(s)
        return conflicts

    def is_valid(seq):
        return get_conflicts(seq) == 0

    # Initial state: Greedy result from seed
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
        if time.time() > deadline - 5:
            break
            
    best_seq = values[:]
    best_len = len(best_seq)
    
    # Simulated Annealing
    current_seq = best_seq[:]
    current_conflicts = get_conflicts(current_seq)
    current_energy = -len(current_seq) * 1000 + current_conflicts
    
    T = 100.0
    T_min = 0.1
    alpha = 0.99
    steps = 20000
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        # 50% chance add, 50% chance remove
        if len(current_seq) < 256 and (len(current_seq) == 0 or np.random.random() < 0.5):
            # Add a random value not in current_seq
            candidates = [x for x in range(1, 257) if x not in current_seq]
            if not candidates:
                continue
            new_val = np.random.choice(candidates)
            # Insert in sorted order
            idx = np.searchsorted(current_seq, new_val)
            new_seq = current_seq[:idx] + [new_val] + current_seq[idx:]
        else:
            if not current_seq:
                continue
            idx = np.random.randint(len(current_seq))
            new_seq = current_seq[:idx] + current_seq[idx+1:]
            
        new_conflicts = get_conflicts(new_seq)
        new_energy = -len(new_seq) * 1000 + new_conflicts
        
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_seq = new_seq
            current_conflicts = new_conflicts
            current_energy = new_energy
            
            if current_conflicts == 0 and len(current_seq) > best_len:
                best_seq = current_seq[:]
                best_len = len(best_seq)
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_seq

# EVOLVE-BLOCK-END
