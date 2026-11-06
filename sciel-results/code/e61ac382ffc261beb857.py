"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from multiprocessing import Pool, cpu_count

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 50.0  # seconds, leave 10s safety margin for 60s wall-clock limit

# Pre-compute initial wire bit-vectors globally for fast access
_INITIAL_WIRES = None

def _get_initial_wires():
    global _INITIAL_WIRES
    if _INITIAL_WIRES is None:
        _INITIAL_WIRES = []
        for i in range(N):
            half = 1 << i
            block = '0' * half + '1' * half
            reps = 1 << (N - i - 1)
            s = block * reps
            v = int(s, 2)
            _INITIAL_WIRES.append(v)
    return _INITIAL_WIRES

# EVOLVE-BLOCK-START
def _verify_worker(network):
    """Worker function for parallel verification."""
    if not network:
        return N <= 1
    
    wires = _get_initial_wires()[:]
    
    for i, j in network:
        wi = wires[i]
        wj = wires[j]
        mask = wi & ~wj
        wires[i] = wi & ~mask
        wires[j] = wj | mask
    
    for i in range(N - 1):
        if wires[i] & ~wires[i + 1]:
            return False
            
    return True

def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses parallel verification and batched greedy removal for efficient search.
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
    if not _verify_worker(current):
        return current
    
    current_count = len(current)
    best_count = current_count
    best_network = current[:]
    
    # Determine number of workers
    n_workers = min(cpu_count(), 8)
    
    # Batch greedy removal: test multiple removals in parallel
    def batch_greedy_remove(net, pool, max_rounds=10):
        """Try to remove comparators by testing batches in parallel."""
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            if len(net) == 0:
                break
            
            # Generate candidate removals
            candidates = []
            for idx in range(len(net)):
                candidate = net[:idx] + net[idx+1:]
                candidates.append((idx, candidate))
            
            if not candidates:
                break
            
            # Verify all candidates in parallel
            candidate_networks = [c[1] for c in candidates]
            results = pool.map(_verify_worker, candidate_networks)
            
            # Find the first successful removal
            found = False
            for idx, result in zip([c[0] for c in candidates], results):
                if result:
                    net = candidates[idx][1]
                    found = True
                    break
            
            if not found:
                break
        
        return net

    # Initial greedy cleanup using parallel verification
    with Pool(n_workers) as pool:
        current = batch_greedy_remove(current, pool)
        current_count = len(current)
        if current_count < best_count:
            best_count = current_count
            best_network = current[:]
        
        # Randomized Hill Climbing with parallel verification
        # We batch multiple move attempts together
        
        while time.time() - start_time < DEADLINE - 5:
            # Generate a batch of candidate moves
            batch_size = min(8, len(current))
            candidates = []
            
            # Strategy A: Removals
            if len(current) > 0:
                for _ in range(batch_size // 2):
                    idx = random.randint(0, len(current) - 1)
                    candidate = current[:idx] + current[idx+1:]
                    candidates.append(('remove', idx, candidate))
            
            # Strategy B: Swaps
            if len(current) >= 2:
                for _ in range(batch_size // 4):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        candidate = current[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        candidates.append(('swap', (i, j), candidate))
            
            # Strategy C: Moves
            if len(current) > 1:
                for _ in range(batch_size // 4):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        candidate = current[:]
                        comp = candidate.pop(i)
                        candidate.insert(j, comp)
                        candidates.append(('move', (i, j), candidate))
            
            if not candidates:
                break
            
            # Verify all candidates in parallel
            candidate_networks = [c[2] for c in candidates]
            results = pool.map(_verify_worker, candidate_networks)
            
            # Process results
            move_success = False
            for (move_type, params, candidate), result in zip(candidates, results):
                if result:
                    current = candidate
                    current_count = len(current)
                    move_success = True
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_network = current[:]
                    
                    # If we found a removal, break and try more removals
                    if move_type == 'remove':
                        break
            
            # If no move succeeded, apply perturbation
            if not move_success:
                # Perturb: Swap 3-5 random pairs
                for _ in range(random.randint(3, 5)):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        current[i], current[j] = current[j], current[i]
                
                # Try to greedily remove from the perturbed state
                perturbed_current = batch_greedy_remove(current, pool, max_rounds=3)
                if len(perturbed_current) < current_count:
                    current = perturbed_current
                    current_count = len(current)
                    if current_count < best_count:
                        best_count = current_count
                        best_network = current[:]

    # Return the best network found
    return best_network


def verify_network_fast(network):
    """
    Verify network using bit-vector simulation with Python integers.
    """
    return _verify_worker(network)


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
