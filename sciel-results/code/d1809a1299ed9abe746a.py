"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all 2^18 Boolean inputs
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    def is_sorted(s):
        """Check if all inputs are sorted (non-decreasing across wires)."""
        # Check if s[:, i] <= s[:, i+1] for all i
        return np.all(s[:, :-1] <= s[:, 1:])

    def apply_comparator(s, i, j):
        """Apply compare-exchange on wires i and j in-place on a copy."""
        # We need to be careful with in-place vs copy
        # For validation, we copy the state at the start
        pass

    def validate_network(net):
        """
        Validates if the network sorts all inputs.
        Returns True if sorted, False otherwise.
        """
        state = all_inputs.copy()
        for i, j in net:
            # Apply comparator (i, j)
            # min goes to i, max goes to j
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        
        return is_sorted(state)

    # Generate initial network: Odd-Even Transposition
    # This is a known valid sorting network with N^2/4 comparators
    # For N=18, it has 18 * 9 = 162? No, N/2 * N = 9 * 18 = 162?
    # Odd-even transposition:
    # Phase 0: (0,1), (2,3), ...
    # Phase 1: (1,2), (3,4), ...
    # ...
    # Total comparators: N * (N/2) = 9 * 18 = 162.
    # Wait, the prompt says seed is 153.
    # Let's check the seed logic in the prompt:
    # `[(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]`
    # Phase 0 (even): i=0,2,...,16 -> 9 comparators
    # Phase 1 (odd): i=1,3,...,15 -> 8 comparators
    # Phase 2 (even): 9 comparators
    # ...
    # 9 phases with 9 comps, 9 phases with 8 comps?
    # Range(N) is 18 phases.
    # Phases 0,2,4,6,8,10,12,14,16 (9 phases) have 9 comparators.
    # Phases 1,3,5,7,9,11,13,15,17 (9 phases) have 8 comparators.
    # Total = 9*9 + 9*8 = 81 + 72 = 153.
    
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network is valid (should be)
    if not validate_network(network):
        # Fallback, though unlikely
        return network

    current_size = len(network)
    
    # Phase 1: Greedy Deletion
    # Try to remove each comparator if the network remains valid
    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE - 10:
            break
            
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > DEADLINE - 10:
                break
                
            # Try removing comparator at idx
            test_net = network[:idx] + network[idx+1:]
            if validate_network(test_net):
                network = test_net
                changed = True
                current_size -= 1
                # Restart loop from end as indices shifted
                break

    # Phase 2: Local Search
    # Use random moves to escape local optima
    # Moves:
    # 1. Delete a random comparator
    # 2. Swap two random comparators
    # 3. Replace a random comparator with a random new one
    
    rng = np.random.default_rng(42)
    
    while time.time() < DEADLINE:
        if not network:
            break
            
        move_type = rng.integers(0, 3)
        
        if move_type == 0:
            # Try to delete a random comparator
            idx = rng.integers(0, len(network))
            test_net = network[:idx] + network[idx+1:]
            if validate_network(test_net):
                network = test_net
                current_size -= 1
                
        elif move_type == 1:
            # Try to swap two random comparators
            if len(network) < 2:
                continue
            idx1 = rng.integers(0, len(network))
            idx2 = rng.integers(0, len(network))
            if idx1 == idx2:
                continue
            test_net = network.copy()
            test_net[idx1], test_net[idx2] = test_net[idx2], test_net[idx1]
            if validate_network(test_net):
                network = test_net
                
        else:
            # Try to replace a random comparator with a random new one
            idx = rng.integers(0, len(network))
            # Generate random comparator
            i = rng.integers(0, N)
            j = rng.integers(i + 1, N)
            new_comp = (i, j)
            if network[idx] == new_comp:
                continue
            test_net = network.copy()
            test_net[idx] = new_comp
            if validate_network(test_net):
                network = test_net
                
    return network
# EVOLVE-BLOCK-END
