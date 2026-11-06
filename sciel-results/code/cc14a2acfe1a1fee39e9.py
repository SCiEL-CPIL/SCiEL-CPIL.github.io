"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        # Use doubling
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    # Get a copy of the initial bitsets
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    # Check if sorted:
    # For all k < l, bitsets[k] must be a subset of bitsets[l].
    # i.e., bitsets[k] & ~bitsets[l] == 0.
    # It is sufficient to check adjacent wires: k and k+1.
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    """
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        # Instead of slicing, we can temporarily remove, check, and restore
        # But slicing is fine for small N.
        
        saved = current[i]
        current.pop(i)
        
        if _is_sorting_network(current):
            # Keep it removed
            pass
        else:
            # Restore
            current.insert(i, saved)
            i += 1
            
    return current

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    Moves: Remove, Add, Swap.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters
    # Higher T allows more exploration
    T = 5.0
    cooling_rate = 0.995
    
    # Pre-compute valid ranges
    max_idx = N - 1
    
    while time.time() < deadline and T > 0.1:
        # Generate a move
        r = random.random()
        neighbor = list(current)
        
        if r < 0.5:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
            else:
                continue
        elif r < 0.8:
            # Try Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
        else:
            # Try Swap
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                else:
                    continue
            else:
                continue
        
        # Check validity
        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                current = neighbor
            elif delta == 0:
                # Neutral
                current = neighbor
            else:
                # Worse (Add)
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-1.0 / T):
                    current = neighbor
        
        # Cool down
        T *= cooling_rate
        
    return best

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best = list(seed_network)
    start_time = time.time()
    # Use 85 seconds out of 90 to be safe
    deadline = start_time + 85
    
    # 1. Initial Greedy reduction to get a compact baseline
    best = _greedy_reduce(best, deadline)
    
    # 2. Simulated Annealing loops
    # We will run SA multiple times.
    
    while time.time() < deadline - 5: # Leave 5s for final cleanup
        # Decide starting point for SA
        if random.random() < 0.7:
            # Local search: start from best
            start_net = list(best)
        else:
            # Global search: start from seed, perturb, reduce
            start_net = list(seed_network)
            # Add some random noise to break symmetry
            num_noise = random.randint(5, 15)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            
            # Reduce the perturbed seed to make it a reasonable starting point
            reduce_deadline = min(deadline, time.time() + 5)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline - 5:
                break
        
        # Run SA
        # Give SA a specific time chunk, e.g., 10s, or run until T drops?
        # Running until T drops might take too long if T starts high.
        # Let's run SA with a sub-deadline.
        sa_deadline = min(deadline - 5, time.time() + 10)
        candidate = _simulated_annealing(start_net, sa_deadline)
        
        if len(candidate) < len(best):
            best = candidate
            # Run a quick greedy reduce on the new best immediately
            if time.time() < deadline - 5:
                quick_deadline = min(deadline - 5, time.time() + 2)
                best = _greedy_reduce(best, quick_deadline)
                
    # Final greedy reduction with remaining time
    if time.time() < deadline:
        best = _greedy_reduce(best, deadline)
        
    return best
# EVOLVE-BLOCK-END
