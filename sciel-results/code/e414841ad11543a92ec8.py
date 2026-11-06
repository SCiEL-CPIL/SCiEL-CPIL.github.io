"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
NUM_BOOL_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START

def _simulate_network(network, inputs):
    """
    inputs: shape (num_inputs, N) array of 0/1
    network: list of (i, j)
    Returns sorted boolean matrix if network sorts all inputs.
    """
    state = inputs.copy()
    for i, j in network:
        # min goes to i, max goes to j
        # For boolean: min(a,b) = a & b, max(a,b) = a | b
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    return state

def _is_sorted_network(network):
    """Check if network sorts all 2^N boolean inputs."""
    # Generate all boolean inputs
    # Using numpy for speed
    # inputs[k, i] is the i-th bit of k
    inputs = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        # bit i of k
        # k ranges 0..2^N-1
        # bit i is (k >> i) & 1
        # Vectorized:
        inputs[:, i] = (np.arange(NUM_BOOL_INPUTS) >> i) & 1
    
    sorted_state = _simulate_network(network, inputs)
    
    # Check if each row is sorted (non-increasing? No, min to i, max to j.
    # Usually sorting networks sort in ascending or descending order.
    # The task says "min to wire i and max to j".
    # If we assume standard sorting where wire 0 is smallest, wire N-1 is largest.
    # Then min goes to the lower index.
    # So wire 0 should have the most 0s (smallest values), wire N-1 most 1s.
    # Wait, for boolean inputs, "sorted" means all 0s come before all 1s.
    # So for a specific input with k ones, the first N-k wires should be 0, and last k wires should be 1.
    
    # Let's check if each row is of the form 0...01...1
    # This is equivalent to checking if the row is non-decreasing.
    # diff = np.diff(sorted_state, axis=1)
    # All diffs should be >= 0.
    
    # Faster check:
    # For each row, count the number of 1s.
    # The row should be 0 for indices 0..N-k-1 and 1 for indices N-k..N-1.
    # Let's just check non-decreasing order.
    diffs = np.diff(sorted_state, axis=1)
    return np.all(diffs >= 0)

def _count_comparators(network):
    return len(network)

def _depth_of_network(network):
    """Calculate parallel depth (longest path in dependency graph)."""
    if not network:
        return 0
    # This is an approximation or exact?
    # Depth is the minimum number of steps if we can run independent comparators in parallel.
    # This is equivalent to the length of the longest chain of comparators where each depends on the previous.
    # We can model this as a longest path in a DAG if we consider time steps.
    # However, a simple greedy layering is often used for reporting, but exact depth requires solving a scheduling problem.
    # For the purpose of this task, depth is reported separately, score is count.
    # I will implement a simple estimate or skip if not needed for scoring.
    # The prompt says "parallel depth is reported separately".
    # I'll just return 0 or a simple estimate to satisfy interface if needed, but score is count.
    return 0

def _try_remove(network, idx):
    """Try removing comparator at idx. Returns True if still valid."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorted_network(new_network)

def _try_swap(network, i, j):
    """Try swapping comparators i and j. Returns True if still valid."""
    if i == j: return True
    new_network = network[:]
    new_network[i], new_network[j] = new_network[j], new_network[i]
    return _is_sorted_network(new_network)

def _try_move(network, i, j):
    """Move comparator from i to j (shifting others)."""
    if i == j: return True
    new_network = network[:]
    comp = new_network.pop(i)
    new_network.insert(j, comp)
    return _is_sorted_network(new_network)

def _greedy_remove(network):
    """Repeatedly remove any removable comparator."""
    changed = True
    while changed:
        changed = False
        for i in range(len(network)):
            if _try_remove(network, i):
                network.pop(i)
                changed = True
                break
    return network

def _seed_odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Start with a known valid network
    # Odd-even transposition is valid but large (153 comparators for N=18? 
    # Actually N phases * (N-1)/2 avg? 
    # Phase 0: 0,2.. 9 comps
    # Phase 1: 1,3.. 8 comps
    # ...
    # Total for N=18: 9*10 = 90? 
    # Let's check: range(0, 17, 2) -> 9 items. range(1, 17, 2) -> 8 items.
    # 18 phases. 9 phases of 9, 9 phases of 8. 81 + 72 = 153. Correct.
    
    network = _seed_odd_even_transposition()
    best_network = network[:]
    
    # Phase 1: Greedy Removal
    # This is fast and gets us close to a local minimum.
    while time.time() < deadline:
        network = _greedy_remove(network)
        if len(network) < len(best_network):
            best_network = network[:]
        
        # If we can't remove anything anymore, break to SA
        # _greedy_remove returns when no single removal works.
        # So if the length didn't change in the last pass, we are at a local min for removal.
        # Actually _greedy_remove loops until no change. So it returns a local min.
        # We need a way to escape.
        
        # Check if we made progress in the last full pass?
        # _greedy_remove is a single function call that runs to completion.
        # So if we call it again immediately, it will do nothing.
        # We need to perturb.
        
        # Let's do a simple Simulated Annealing or Random Perturbation loop.
        break # Exit greedy phase, move to SA
    
    # Phase 2: Simulated Annealing / Local Search
    # We accept worse solutions with some probability to escape local minima.
    # Or we can just do random swaps/moves and accept if better.
    # Given the time limit, a simple hill-climbing with random restarts or SA is good.
    
    current_network = best_network[:]
    current_size = len(current_network)
    best_size = current_size
    
    # Temperature for SA
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate a neighbor
        # Options:
        # 1. Add a random comparator
        # 2. Remove a random comparator (if valid)
        # 3. Swap two comparators
        # 4. Move a comparator
        
        move_type = random.random()
        
        new_network = current_network[:]
        
        if move_type < 0.3:
            # Add random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_network.append((i, j))
        elif move_type < 0.6:
            # Remove random comparator
            if len(new_network) > 0:
                idx = random.randint(0, len(new_network) - 1)
                new_network.pop(idx)
        elif move_type < 0.8:
            # Swap
            if len(new_network) > 1:
                i = random.randint(0, len(new_network) - 1)
                j = random.randint(0, len(new_network) - 1)
                new_network[i], new_network[j] = new_network[j], new_network[i]
        else:
            # Move
            if len(new_network) > 1:
                i = random.randint(0, len(new_network) - 1)
                j = random.randint(0, len(new_network) - 1)
                if i != j:
                    comp = new_network.pop(i)
                    new_network.insert(j, comp)
        
        # Check validity
        # To save time, we can do a quick check or full check.
        # Full check is ~262k * depth. Depth ~10-20. ~5M ops.
        # In Python/Numpy this is fast (<1ms).
        # So we can afford many checks.
        
        if _is_sorted_network(new_network):
            new_size = len(new_network)
            delta = new_size - current_size
            
            # Acceptance criterion
            if delta < 0:
                # Better
                current_network = new_network
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_network = current_network[:]
            else:
                # Worse or equal
                # Accept with probability exp(-delta / T)
                # delta is positive (worse)
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_network = new_network
                    current_size = new_size
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    # Final greedy removal on the best found
    best_network = _greedy_remove(best_network)
    
    return best_network

# EVOLVE-BLOCK-END
