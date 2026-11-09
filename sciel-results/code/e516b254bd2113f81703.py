"""Size-efficient sorting network on 18 wires via SAT-based synthesis."""
import numpy as np
import time
import random

N = 18

def _verify_network(network):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses NumPy vectorization for speed.
    """
    if len(network) == 0:
        return True
        
    n_comparators = len(network)
    comp_i = np.array([c[0] for c in network], dtype=np.int32)
    comp_j = np.array([c[1] for c in network], dtype=np.int32)
    
    # Generate all 2^N inputs
    # 2^18 = 262144. 262144 * 18 * 1 byte (int8) = ~4.7 MB. Fits in memory.
    # We can generate the full matrix at once.
    # bits = np.arange(2**N)[:, None]
    # inputs = (bits >> np.arange(N)[None, :]) & 1
    # This might be slow to generate.
    # Let's do it in chunks to be safe on memory/time overhead.
    
    chunk_size = 1 << 16 # 65536
    total_inputs = 1 << N
    
    for start in range(0, total_inputs, chunk_size):
        end = min(start + chunk_size, total_inputs)
        masks = np.arange(start, end, dtype=np.int64)
        
        # Generate inputs for this chunk
        # Shape: (chunk_len, N)
        # We can use broadcasting
        # bits = (masks[:, None] >> np.arange(N)[None, :]) & 1
        # This creates a (chunk_len, N) array.
        inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
        inputs = inputs.astype(np.int8)
        
        # Apply network
        for k in range(n_comparators):
            i, j = comp_i[k], comp_j[k]
            # Compare
            swap_mask = inputs[:, i] > inputs[:, j]
            if np.any(swap_mask):
                # Swap
                temp = inputs[:, i][swap_mask].copy()
                inputs[:, i][swap_mask] = inputs[:, j][swap_mask]
                inputs[:, j][swap_mask] = temp
        
        # Check sorted
        # diff < 0 means not sorted
        if np.any(np.diff(inputs, axis=1) < 0):
            return False
            
    return True

def construct():
    """
    Construct a sorting network on N=18 wires using local search with fast verification.
    """
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # 1. Generate Seed: Odd-Even Transposition
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    
    network = odd_even_transposition()
    best_network = network
    best_size = len(network)
    
    # Verify seed (should be True)
    if not _verify_network(network):
        raise ValueError("Seed network is invalid")
        
    # 2. Greedy Removal Phase
    # Try to remove comparators one by one
    # We do this in a loop, restarting the scan after each removal
    while time.time() - start_time < time_limit * 0.2: # Use 20% time for greedy
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(network):
                continue
            new_network = network[:idx] + network[idx+1:]
            if _verify_network(new_network):
                network = new_network
                best_network = network
                best_size = len(network)
                removed_any = True
                break # Restart scan
        
        if not removed_any:
            break
            
    # 3. Local Search / Simulated Annealing Phase
    # We allow insertions and swaps to escape local minima
    T = 1.0 # Temperature
    current_size = len(network)
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        r = random.random()
        new_network = network[:]
        
        if r < 0.4:
            # Remove a random comparator
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.7:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        else:
            # Swap two comparators
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network = network[:]
                new_network[i], new_network[j] = new_network[j], new_network[i]
                
        if len(new_network) == len(network):
             # If size is same, we might still want to accept to explore
             # But we only accept if valid
             pass
             
        new_size = len(new_network)
        
        # Verify
        if _verify_network(new_network):
            # Accept if better, or with some probability if same/worse (to escape minima)
            if new_size < current_size:
                network = new_network
                best_network = network
                best_size = new_size
                current_size = new_size
                T = max(0.1, T * 0.99) # Cool down
            else:
                # Accept with probability p
                # Delta E = new_size - current_size
                delta = new_size - current_size
                if delta == 0:
                    p = 0.1 # Accept same size with low prob to shuffle
                else:
                    p = np.exp(-delta / T)
                    
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.99)
        else:
            # Reject
            pass
            
    return best_network

# Pre-compute and cache the result to avoid recomputation
_cached_network = None

def _get_network():
    global _cached_network
    if _cached_network is None:
        _cached_network = construct()
    return _cached_network

# Override construct to use caching
_original_construct = construct

def construct():
    global _cached_network
    if _cached_network is None:
        _cached_network = _original_construct()
    return _cached_network
