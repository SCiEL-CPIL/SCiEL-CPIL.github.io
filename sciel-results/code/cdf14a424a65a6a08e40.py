"""Size-efficient sorting network search on 16 wires."""
import time
import random
import numpy as np

N = 16
DEADLINE = time.time() + 80.0  # 80s safety margin (90s limit)

def fast_verify(network):
    """Check all 65536 Boolean inputs using NumPy bit-parallel simulation."""
    if not network:
        return True
    n = len(network)
    # Generate all 65536 inputs as a 2D array: shape (65536, N)
    # bits[k, i] = (k >> i) & 1
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = ((np.arange(1 << N) >> i) & 1).astype(np.uint8)
    
    # Simulate each comparator
    for i, j in network:
        # For each input row, compare wires i and j
        a = inputs[:, i]
        b = inputs[:, j]
        # min goes to wire i, max goes to wire j
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check if all rows are sorted: inputs[:, k] <= inputs[:, k+1] for all k
    for k in range(N - 1):
        if not np.all(inputs[:, k] <= inputs[:, k + 1]):
            return False
    return True

def verify_network(network):
    """Check all 65536 Boolean inputs using zero-one principle."""
    return fast_verify(network)

def odd_even_network():
    """Odd-even transposition sorting network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def bitonic_sorter(n):
    """Generate bitonic sorting network for n wires (power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half ascending, second half descending
    net = []
    # Sort first half
    net.extend(bitonic_sorter(half))
    # Sort second half (reversed)
    for i, j in bitonic_sorter(half):
        net.append((n - 1 - j, n - 1 - i))
    
    # Merge phase
    for k in range(1, n):
        for i in range(0, n - k):
            if i + k < n:
                net.append((i, i + k))
    return net

def try_remove(network, idx):
    """Try removing comparator at idx, return new network if still valid."""
    if len(network) <= 1:
        return None
    new_net = network[:idx] + network[idx+1:]
    if verify_network(new_net):
        return new_net
    return None

def compute_depth(network):
    """Compute parallel depth (longest path) of the network."""
    wire_time = [0] * N
    for i, j in network:
        t = max(wire_time[i], wire_time[j]) + 1
        wire_time[i] = t
        wire_time[j] = t
    return max(wire_time)

def random_network(num_comps):
    """Generate a random network (likely invalid, but used for diversity)."""
    comps = []
    for _ in range(num_comps):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps

def local_search_starting_from(network, max_time=5.0):
    """Iterative improvement: try to remove redundant comparators."""
    best = list(network)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        indices = list(range(len(best)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > DEADLINE:
                return best
            new_net = try_remove(best, idx)
            if new_net is not None:
                best = new_net
                improved = True
                break
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Try multiple starting points
    candidates = [
        odd_even_network(),
        bitonic_sorter(N)
    ]
    
    # Pick the best (smallest) valid starting point
    best = None
    for cand in candidates:
        if verify_network(cand):
            if best is None or len(cand) < len(best):
                best = cand
    
    if best is None:
        best = odd_even_network()
    
    best_count = len(best)
    
    # Phase 1: Local search to remove redundant comparators
    while time.time() < DEADLINE - 5:  # Leave 5s for final verification
        improved = False
        # Try removing each comparator in random order
        indices = list(range(len(best)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > DEADLINE:
                break
            new_net = try_remove(best, idx)
            if new_net is not None:
                best = new_net
                best_count = len(best)
                improved = True
                break  # Restart scan with new length
        
        if not improved:
            # Stuck at local minimum - try perturbation
            # Add a random comparator, then try to remove others
            perturbed = False
            for _ in range(10):
                if time.time() > DEADLINE:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(best))
                candidate = best[:pos] + [(i, j)] + best[pos:]
                if verify_network(candidate):
                    # Now try to remove from this larger network
                    new_best = list(candidate)
                    new_count = len(new_best)
                    for _ in range(new_count):
                        if time.time() > DEADLINE:
                            break
                        rem_indices = list(range(len(new_best)))
                        random.shuffle(rem_indices)
                        for rem_idx in rem_indices:
                            if time.time() > DEADLINE:
                                break
                            test_net = new_best[:rem_idx] + new_best[rem_idx+1:]
                            if verify_network(test_net):
                                new_best = test_net
                                new_count -= 1
                                break
                    if len(new_best) < best_count:
                        best = new_best
                        best_count = len(best)
                        perturbed = True
                        improved = True
                        break
            if not perturbed and time.time() > start_time + 30:
                # Try fresh random restarts
                found = False
                for _ in range(20):
                    if time.time() > DEADLINE:
                        break
                    n_comps = random.randint(45, best_count - 1)
                    net = random_network(n_comps)
                    if verify_network(net):
                        best = net
                        best_count = len(best)
                        found = True
                        break
                if not found:
                    break
    
    # Final verification
    if not verify_network(best):
        best = odd_even_network()
    
    return best
# EVOLVE-BLOCK-END
