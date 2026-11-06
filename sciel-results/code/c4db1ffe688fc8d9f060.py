import numpy as np
import time
import random

N = 16
TOTAL_INPUTS = 1 << N
MASK_ALL = TOTAL_INPUTS - 1

# Precompute sorted status for all possible masks
# A mask is sorted if it is of the form 0...01...1
# This is equivalent to: inv = (~mask) & MASK_ALL, and (inv & (inv + 1)) == 0
def _get_sorted_mask():
    masks = np.arange(TOTAL_INPUTS, dtype=np.int32)
    inv = (~masks) & MASK_ALL
    return (inv & (inv + 1)) == 0

SORTED_MASK = _get_sorted_mask()

def _simulate_network(network):
    """
    Simulate the network on all 65536 inputs.
    Returns the number of unsorted outputs.
    """
    if not network:
        # Empty network: only mask 0 and mask MASK_ALL are sorted? No, 
        # if input is not sorted, output is not sorted.
        # Only inputs that are already sorted remain sorted.
        # But we need to check how many of the 65536 inputs are sorted initially.
        # There are N+1 = 17 sorted inputs.
        return TOTAL_INPUTS - np.sum(SORTED_MASK)

    masks = np.arange(TOTAL_INPUTS, dtype=np.int32)
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j)
        if np.any(swap):
            # Apply swap to all masks where swap is True
            # Clear bit i, set bit j
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    
    # Check how many are sorted
    unsorted_count = TOTAL_INPUTS - np.sum(SORTED_MASK & _is_sorted_array(masks))
    return unsorted_count

def _is_sorted_array(masks):
    """
    Check which masks in the array are sorted.
    Returns a boolean array of the same shape as masks.
    """
    inv = (~masks) & MASK_ALL
    return (inv & (inv + 1)) == 0

def _cost(network):
    """
    Calculate cost: 1000 * unsorted_count + len(network)
    """
    unsorted = _simulate_network(network)
    return 1000 * unsorted + len(network)

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _greedy_remove(network):
    """
    Greedily remove redundant comparators.
    """
    improved = True
    while improved:
        improved = False
        n_comps = len(network)
        for idx in range(n_comps):
            new_network = network[:idx] + network[idx+1:]
            if _simulate_network(new_network) == 0:
                network = new_network
                improved = True
                break
    return network

def construct():
    deadline = time.time() + 58  # Safety margin
    rng = random.Random(42)
    
    # 1. Start with Batcher's 63
    network = _batcher(N)
    
    # 2. Greedy removal to get to a local minimum (likely 60)
    network = _greedy_remove(network)
    best_network = network[:]
    best_cost = _cost(network)
    
    # 3. Simulated Annealing
    # T starts high to allow invalid moves, decays
    T = 100.0
    T_min = 0.1
    alpha = 0.995
    
    # We want to minimize cost.
    # Cost = 1000 * unsorted + count
    
    while time.time() < deadline and T > T_min:
        current_cost = _cost(network)
        
        # Generate a neighbor
        move_type = rng.random()
        new_network = network[:]
        
        if move_type < 0.4:
            # Swap two comparators
            if len(network) > 1:
                i, j = rng.sample(range(len(network)), 2)
                new_network[i], new_network[j] = new_network[j], new_network[i]
            else:
                continue
        elif move_type < 0.7:
            # Move a comparator to a random position
            if len(network) > 1:
                i = rng.randint(0, len(network) - 1)
                j = rng.randint(0, len(network) - 1)
                comp = new_network.pop(i)
                new_network.insert(j, comp)
            else:
                continue
        elif move_type < 0.85:
            # Remove a random comparator
            if len(network) > 0:
                i = rng.randint(0, len(network) - 1)
                new_network.pop(i)
            else:
                continue
        else:
            # Add a random comparator
            # Pick random i, j
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            pos = rng.randint(0, len(new_network))
            new_network.insert(pos, (i, j))
            
        new_cost = _cost(new_network)
        
        # Acceptance criterion
        if new_cost < current_cost:
            network = new_network
            if new_cost < best_cost:
                best_cost = new_cost
                best_network = network[:]
        else:
            delta = new_cost - current_cost
            if delta == 0:
                network = new_network
            else:
                p = np.exp(-delta / T)
                if rng.random() < p:
                    network = new_network
        
        T *= alpha
        
    # 4. Final greedy removal on the best network found
    best_network = _greedy_remove(best_network)
    
    # One last check to see if we can remove any more from the final result
    # (greedy_remove already does this, but just in case)
    return best_network

# Helper for debugging, not used in construct
if __name__ == "__main__":
    net = construct()
    print(f"Comparators: {len(net)}")
    print(f"Valid: {_simulate_network(net) == 0}")
