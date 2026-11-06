"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a SAT-inspired heuristic approach.
    We use a layered structure with fixed depth and optimize comparator placement
    using a greedy + local search strategy guided by the 0-1 principle.
    
    The approach:
    1. Start with a known good network (odd-even transposition).
    2. Use local search to remove comparators where possible while maintaining correctness.
    3. Use the 0-1 principle: a network sorts all inputs iff it sorts all 2^N boolean inputs.
    
    For efficiency within 60s, we use a practical heuristic:
    - Start from a dense layered network
    - Iteratively try removing each comparator and check if sorting is still guaranteed
    - Use random restarts and simulated annealing to escape local minima
    """
    import random
    import time
    
    start_time = time.time()
    timeout = 55  # seconds, leave buffer
    
    def apply_compare(network_state, i, j):
        """Apply a compare-exchange to a state vector."""
        if network_state[i] > network_state[j]:
            network_state[i], network_state[j] = network_state[j], network_state[i]
    
    def is_sorting_network(comparators):
        """Check if the network sorts all boolean inputs using the 0-1 principle."""
        # For efficiency, we can use a smarter check:
        # A network sorts all inputs iff for every permutation of distinct values,
        # the output is sorted. But with 0-1 principle, we only need boolean inputs.
        # However, checking all 2^18 = 262144 inputs is feasible.
        
        n = len(comparators)
        if n == 0:
            return True
            
        # Check all boolean inputs
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            for (i, j) in comparators:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check if sorted
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True
    
    def is_sorting_network_fast(comparators):
        """
        Faster check using the fact that we only need to verify the network 
        produces sorted output for all boolean inputs.
        We can use a more efficient representation.
        """
        if not comparators:
            return True
            
        # Use numpy for speed
        for mask in range(1 << N):
            state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
            for (i, j) in comparators:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            if not np.all(state[:-1] <= state[1:]):
                return False
        return True
    
    def is_sorting_network_v2(comparators):
        """
        Even faster: use the observation that for boolean inputs,
        the number of 1s is preserved. So we can group by number of 1s.
        For each k (number of 1s), check all C(N,k) inputs.
        Total is still 2^N but we can short-circuit.
        """
        if not comparators:
            return True
            
        # Precompute comparator list as tuples for speed
        comp_list = list(comparators)
        
        for mask in range(1 << N):
            state = [0] * N
            for i in range(N):
                state[i] = (mask >> i) & 1
            
            for (i, j) in comp_list:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            
            # Check sorted
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True
    
    # Start with odd-even transposition network
    def odd_even_transposition():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    current = odd_even_transposition()
    best = list(current)
    best_len = len(current)
    
    # Local search: try to remove comparators
    def try_remove(comparators, index):
        """Try removing comparator at index and check if still sorting."""
        new_comps = comparators[:index] + comparators[index+1:]
        return new_comps, is_sorting_network_v2(new_comps)
    
    # Greedy removal
    random.seed(42)
    iteration = 0
    while time.time() - start_time < timeout:
        iteration += 1
        # Shuffle order of removal attempts
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            new_comps, valid = try_remove(current, idx)
            if valid:
                current = new_comps
                removed_any = True
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        if not removed_any:
            # Try adding comparators at random positions to escape local minima
            if len(current) < 200 and random.random() < 0.3:
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                pos = random.randint(0, len(current))
                new_comps = current[:pos] + [(i, j)] + current[pos:]
                if is_sorting_network_v2(new_comps):
                    current = new_comps
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
            else:
                # Reset to best and try again with different random seed
                current = list(best)
    
    return best
# EVOLVE-BLOCK-END
