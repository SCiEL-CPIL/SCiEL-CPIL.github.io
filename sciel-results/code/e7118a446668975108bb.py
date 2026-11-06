"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 17
NUM_INPUTS = 1 << N  # 131072

# EVOLVE-BLOCK-START

def _apply_network(network, bits):
    """Apply the network to a bitmask. Returns sorted bitmask (0s then 1s)."""
    for i, j in network:
        bi = (bits >> i) & 1
        bj = (bits >> j) & 1
        if bi > bj:
            bits ^= (1 << i) | (1 << j)
    return bits

def _is_sorted_mask(mask):
    """Check if the mask is sorted (all 1s first, then 0s)."""
    if mask == 0:
        return True
    # Count the number of 1s
    num_ones = bin(mask).count('1')
    # The sorted version should be (1 << num_ones) - 1
    return mask == (1 << num_ones) - 1

def _verify_network(network, sample_size=None, seed=42):
    """Verify the network sorts all inputs (or a sample)."""
    if sample_size is None or sample_size >= NUM_INPUTS:
        for bits in range(NUM_INPUTS):
            result = _apply_network(network, bits)
            if not _is_sorted_mask(result):
                return False
        return True
    else:
        rng = random.Random(seed)
        for _ in range(sample_size):
            bits = rng.randint(0, NUM_INPUTS - 1)
            result = _apply_network(network, bits)
            if not _is_sorted_mask(result):
                return False
        return True

def _count_comparators(network):
    return len(network)

def _odd_even_transposition_network(n):
    """Generate odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _batcher_odd_merge_sort(n):
    """Generate a Batcher odd-even merge sort network."""
    network = []
    
    def batcher(n, start, end):
        """Batcher's odd-even merge sort for wires [start, end)."""
        length = end - start
        if length <= 1:
            return
        if length == 2:
            network.append((start, start + 1))
            return
        # Split into two halves
        mid = (start + end) // 2
        # Recursively sort each half
        batcher(start, mid)
        batcher(mid, end)
        # Merge the two sorted halves using odd-even merge
        _odd_even_merge(start, mid, mid, end, network)
    
    def _odd_even_merge(a_start, a_end, b_start, b_end, net):
        """Odd-even merge of two sorted sequences."""
        # This is a simplified version; for correctness we use a different approach
        pass
    
    # Use a simpler recursive Batcher network
    network = []
    def _batcher_recursive(n):
        """Generate Batcher odd-even merge sort network for n wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into two halves
        half = (n + 1) // 2
        left = _batcher_recursive(half)
        right = _batcher_recursive(n - half)
        
        # Shift right half by 'half'
        right_shifted = [(i + half, j + half) for i, j in right]
        
        # Merge using odd-even merge
        merge = _odd_even_merge_network(half, n - half)
        
        return left + right_shifted + merge
    
    def _odd_even_merge_network(n1, n2):
        """Generate odd-even merge network for merging two sorted sequences."""
        # For simplicity, use a comparison network that works
        # This is a simplified approach
        result = []
        # We need to properly implement odd-even merge
        # For now, use a simple merge that may not be optimal but is correct
        # Actually, let's use a different approach: just use the known good networks
        
        # Simple approach: create a network that compares all cross pairs
        # This is O(n^2) but correct
        for i in range(n1):
            for j in range(n2):
                result.append((i, n1 + j))
        return result
    
    return _batcher_recursive(n)

def _known_small_networks():
    """Return known good networks for small N."""
    # For N=17, we'll use a hybrid approach
    # Start with a reasonable baseline
    return None

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = start_time + 70  # 70 second deadline (90s limit, 20s safety margin)
    
    # Start with odd-even transposition (136 comparators)
    best_network = _odd_even_transposition_network(N)
    best_count = _count_comparators(best_network)
    
    # Try to improve using random search / local optimization
    rng = random.Random(12345)
    
    # First, try a few different initial networks
    initial_networks = [best_network]
    
    # Try a simple merge sort network
    merge_network = _batcher_odd_merge_sort(N)
    if merge_network and _verify_network(merge_network, sample_size=10000, seed=42):
        if len(merge_network) < best_count:
            best_network = merge_network
            best_count = len(merge_network)
    
    # Local search: try to remove redundant comparators
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Try removing each comparator and checking validity
        improved = False
        network_copy = list(best_network)
        
        for idx in range(len(network_copy) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            candidate = network_copy[:idx] + network_copy[idx+1:]
            
            # Quick check with sample
            if _verify_network(candidate, sample_size=5000, seed=idx):
                # Full verification
                if _verify_network(candidate):
                    network_copy = candidate
                    best_count -= 1
                    improved = True
                    break
        
        if not improved:
            break
        
        best_network = network_copy
    
    # Try adding comparators in different positions to enable more removals
    # This is a more advanced optimization
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create a perturbed version
        network_copy = list(best_network)
        
        # Add a random comparator
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        pos = rng.randint(0, len(network_copy))
        network_copy.insert(pos, (i, j))
        
        # Try to remove comparators from this new network
        improved = False
        for idx in range(len(network_copy) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate = network_copy[:idx] + network_copy[idx+1:]
            
            if _verify_network(candidate, sample_size=5000, seed=idx):
                if _verify_network(candidate):
                    network_copy = candidate
                    improved = True
                    break
        
        if improved and len(network_copy) < best_count:
            best_network = network_copy
            best_count = len(network_copy)
    
    return best_network

# EVOLVE-BLOCK-END
