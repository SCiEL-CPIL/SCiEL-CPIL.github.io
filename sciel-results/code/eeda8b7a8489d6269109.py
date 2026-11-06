"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, leave 5s safety margin for 90s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses a hybrid approach:
    1. Start with a good baseline (odd-even transposition).
    2. Aggressive greedy removal to get close to local optimum.
    3. Simulated Annealing to escape local optima (accepting worse solutions).
    4. Final greedy polish.
    """
    start_time = time.time()
    
    # Baseline: odd-even transposition
    def odd_even_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with the odd-even network
    current = odd_even_network(N)
    
    # Verify it works (should always work)
    if not verify_network_fast(current):
        return current
    
    best_count = len(current)
    best_network = current[:]
    
    # Phase 1: Aggressive greedy removal
    current = greedy_remove(current, start_time)
    if len(current) < best_count:
        best_count = len(current)
        best_network = current[:]
    
    # Phase 2: Simulated Annealing to escape local optima
    current = simulated_annealing(current, best_network, best_count, start_time)
    
    # Final polish: try to remove any remaining redundant comparators
    current = greedy_remove(current, start_time)
    if len(current) < best_count:
        best_count = len(current)
        best_network = current[:]
    
    return best_network


def greedy_remove(network, start_time):
    """
    Greedy removal: repeatedly try to remove comparators from the network.
    Uses bit-parallel verification for speed.
    """
    current = network[:]
    
    # Multiple passes to ensure we find all removable comparators
    for pass_num in range(15):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try removing from the end first (often easier to remove)
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                return current
            
            candidate = current[:idx] + current[idx+1:]
            
            if verify_network_fast(candidate):
                current = candidate
                improved = True
                # Restart from the end after removal
                break
        
        if not improved:
            # Try from the beginning
            for idx in range(len(current)):
                if time.time() - start_time > DEADLINE:
                    return current
                
                candidate = current[:idx] + current[idx+1:]
                
                if verify_network_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if not improved:
                break
    
    return current


def simulated_annealing(initial, best_network, best_count, start_time):
    """
    Simulated Annealing to escape local optima.
    Accepts worse solutions with a probability that decreases over time.
    """
    current = initial[:]
    current_count = len(current)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    T = T_start
    
    # Track best
    best_in_sa = best_count
    best_net_sa = best_network[:]
    
    max_iterations = 2000
    
    for iter_num in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            return best_net_sa
        
        # Generate a neighbor
        neighbor = mutate_network(current)
        
        if len(neighbor) > MAX_COMPARATORS:
            T *= cooling_rate
            continue
        
        # Verify the neighbor
        if verify_network_fast(neighbor):
            neighbor_count = len(neighbor)
            
            # Accept if better or equal
            if neighbor_count <= current_count:
                current = neighbor
                current_count = neighbor_count
                
                if current_count < best_in_sa:
                    best_in_sa = current_count
                    best_net_sa = current[:]
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = neighbor_count - current_count
                if delta < 0:
                    delta = 0
                prob = np.exp(-delta / max(T, 1e-6))
                if random.random() < prob:
                    current = neighbor
                    current_count = neighbor_count
        
        # Cool down
        T *= cooling_rate
        
        # If temperature is too low, restart with a random kick
        if T < T_end:
            # Try a more aggressive restart: remove a random chunk and re-add
            if len(current) > 50:
                # Remove a random comparator and add a different one
                idx = random.randint(0, len(current) - 1)
                current = current[:idx] + current[idx+1:]
                if verify_network_fast(current):
                    current_count = len(current)
                    if current_count < best_in_sa:
                        best_in_sa = current_count
                        best_net_sa = current[:]
            
            # Reset temperature
            T = T_start * 0.5
    
    return best_net_sa


def mutate_network(network):
    """
    Apply a random mutation to the network.
    """
    if len(network) == 0:
        return network[:]
    
    candidate = network[:]
    mutation_type = random.random()
    
    if mutation_type < 0.4:
        # Remove a random comparator
        idx = random.randint(0, len(candidate) - 1)
        candidate = candidate[:idx] + candidate[idx+1:]
    elif mutation_type < 0.6:
        # Add a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(candidate))
        candidate.insert(pos, (i, j))
    elif mutation_type < 0.8:
        # Replace a random comparator
        idx = random.randint(0, len(candidate) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate[idx] = (i, j)
    elif mutation_type < 0.9:
        # Move a comparator to a different position
        idx = random.randint(0, len(candidate) - 1)
        comp = candidate[idx]
        candidate.pop(idx)
        new_pos = random.randint(0, len(candidate))
        candidate.insert(new_pos, comp)
    else:
        # Swap two comparators
        if len(candidate) >= 2:
            idx1 = random.randint(0, len(candidate) - 1)
            idx2 = random.randint(0, len(candidate) - 1)
            if idx1 != idx2:
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
    
    return candidate


def verify_network_fast(network):
    """
    Fast verification using bit-parallelism with larger batches.
    """
    n = len(network)
    if n == 0:
        return N <= 1
    
    # Use larger batches for better numpy performance
    batch_size = 4096
    total_inputs = 1 << N
    
    for batch_start in range(0, total_inputs, batch_size):
        batch_end = min(batch_start + batch_size, total_inputs)
        batch = np.arange(batch_start, batch_end, dtype=np.uint32)
        batch_size_actual = batch_end - batch_start
        
        # Initialize wires: shape (batch_size, N)
        wires = np.zeros((batch_size_actual, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (batch >> i) & 1
        
        # Apply each comparator
        for i, j in network:
            # Compare and swap where needed
            mask = wires[:, i] > wires[:, j]
            if np.any(mask):
                # Use temporary array for swap
                temp = wires[mask, i].copy()
                wires[mask, i] = wires[mask, j]
                wires[mask, j] = temp
        
        # Check if sorted: for each row, check if non-decreasing
        for k in range(N - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
    
    return True


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
