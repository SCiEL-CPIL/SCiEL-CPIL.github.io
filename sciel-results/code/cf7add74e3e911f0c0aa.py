"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 80.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Sorted values: bitmask x is sorted (0s then 1s) iff x = 2^N - 2^k for k in [0, N]
    sorted_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.int64)
    
    def is_sorted(inputs):
        """Check if all inputs in the array are sorted (bitmask form)."""
        count = 0
        for s in sorted_values:
            count += np.sum(inputs == s)
        return count == len(inputs)
    
    def apply_network(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks."""
        for (i, j) in comparators:
            mask_i = (inputs >> i) & 1
            mask_j = (inputs >> j) & 1
            swap_mask = mask_j & (~mask_i)
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs
        
    def verify_network(network):
        """Verify if the network sorts all inputs."""
        if not network:
            return False
        test_inputs = all_inputs.copy()
        result = apply_network(test_inputs, network)
        return is_sorted(result)

    # Start with odd-even transposition network
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_network(best_network):
        return best_network

    rng = random.Random(42)
    
    def greedy_remove(network):
        """Try to remove comparators one by one, checking validity."""
        net = list(network)
        changed = True
        # Randomize the order of removal attempts to avoid deterministic local minima
        indices = list(range(len(net)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > DEADLINE - 2:
                break
            # If the index is out of bounds (because we removed things), skip
            if idx >= len(net):
                continue
            
            # Try removing net[idx]
            candidate = net[:idx] + net[idx+1:]
            
            # Quick check on a subset first to save time on invalid candidates
            subset_size = min(1024, 1 << N)
            test_inputs = all_inputs[:subset_size].copy()
            result = apply_network(test_inputs, candidate)
            if is_sorted(result):
                # Full check
                full_result = apply_network(all_inputs.copy(), candidate)
                if is_sorted(full_result):
                    net = candidate
        
        return net

    # Phase 1: Initial Greedy Removal
    best_network = greedy_remove(best_network)
    best_count = len(best_network)
    
    # Phase 2: Simulated Annealing / Random Restarts with Perturbation
    # We will perform random swaps to escape local minima, then try greedy removal again.
    
    # Precompute a set of "hard" inputs for faster rejection if we want, 
    # but full verification is fast enough with NumPy for ~1000 checks/sec.
    
    while time.time() - start_time < DEADLINE - 5:
        # Create a perturbed version of the best network
        perturbed = list(best_network)
        
        # Perform a few random swaps
        num_swaps = rng.randint(1, 5)
        for _ in range(num_swaps):
            if len(perturbed) > 1:
                i = rng.randint(0, len(perturbed) - 2)
                perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
        
        # Check if perturbed is valid
        if not verify_network(perturbed):
            continue
            
        # Try to remove comparators from this valid perturbed network
        reduced = greedy_remove(perturbed)
        
        if len(reduced) < best_count:
            best_network = reduced
            best_count = len(best_network)
            
        # If we hit a new best, we might want to do a more aggressive search around it
        # But for now, just continue loop.

    return best_network
# EVOLVE-BLOCK-END
