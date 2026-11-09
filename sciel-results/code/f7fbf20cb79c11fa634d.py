"""Size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START

def _build_wires():
    """Pre-compute the 2^N boolean input wires as numpy arrays."""
    wires = []
    for i in range(N):
        wires.append((np.arange(NUM_INPUTS) >> i) & 1)
    return wires


def _apply_network(network, wires=None):
    """Apply network to all 2^N boolean inputs. Returns True if all sorted."""
    if wires is None:
        wires = _build_wires()
    
    # Work on a copy to avoid mutation
    w = [w.copy() for w in wires]
    
    for (a, b) in network:
        wa = w[a]
        wb = w[b]
        new_a = np.minimum(wa, wb)
        new_b = np.maximum(wa, wb)
        w[a] = new_a
        w[b] = new_b
    
    # Check if sorted: wire[i] <= wire[i+1] for all i
    for i in range(N - 1):
        if np.any(w[i] > w[i + 1]):
            return False
    return True


def _verify_network(network, wires=None):
    """Verify network sorts all boolean inputs. Returns True if correct."""
    return _apply_network(network, wires)


def _odd_even_transposition(n):
    """Build odd-even transposition network for n wires."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network


def _batcher_odd_even_merge_sort(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    
    network = []
    
    def odd_even_merge(k, offset, direction):
        """Odd-even merge of two sorted sequences of length k starting at offset."""
        if k == 1:
            return
        half = k // 2
        # Recursively merge odd and even positions
        odd_even_merge(half, offset + 1, direction)
        odd_even_merge(half, offset, direction)
        # Compare-exchange at positions offset + i and offset + half + i
        for i in range(half):
            a = offset + i
            b = offset + half + i
            if a < b:
                network.append((a, b))
            else:
                network.append((b, a))
    
    def odd_even_merge_sort(k, offset):
        """Sort k elements starting at offset using odd-even merge sort."""
        if k <= 1:
            return
        half = k // 2
        # Recursively sort first half and second half
        odd_even_merge_sort(half, offset)
        odd_even_merge_sort(k - half, offset + half)
        # Merge
        odd_even_merge(half, offset, 0)
    
    odd_even_merge_sort(n, 0)
    return network


def _bitonic_merge(k, offset, ascending=True):
    """Bitonic merge of k elements starting at offset."""
    if k <= 1:
        return []
    
    comparators = []
    for i in range(k // 2):
        a = offset + i
        b = offset + k - 1 - i
        if ascending:
            comparators.append((a, b))
        else:
            comparators.append((b, a))
    
    half = k // 2
    if half > 1:
        left = _bitonic_merge(half, offset, ascending)
        right = _bitonic_merge(half, offset + half, ascending)
        comparators = left + right + comparators
    
    return comparators


def _bitonic_sort(n):
    """Build bitonic sort network for n wires."""
    if n <= 1:
        return []
    
    network = []
    
    # Build bitonic sequence
    for size in range(2, n + 1):
        if size > n:
            break
        for start in range(0, n, size):
            end = min(start + size, n)
            if end - start < 2:
                continue
            # Determine if this segment should be ascending or descending
            # In bitonic sort, we alternate directions based on the level
            level = 0
            temp_size = size
            while temp_size > 1:
                level += 1
                temp_size //= 2
            
            # For simplicity, let's use a standard bitonic sort construction
            pass
    
    # Actually, let's use a cleaner bitonic sort
    def bitonic_sort_rec(n, offset, ascending):
        if n <= 1:
            return []
        
        comparators = []
        half = n // 2
        if half > 1:
            # Recursively sort first half ascending, second half descending (or vice versa)
            left = bitonic_sort_rec(half, offset, True)
            right = bitonic_sort_rec(n - half, offset + half, False)
            comparators = left + right
            # Merge
            comparators += _bitonic_merge(n, offset, ascending)
        else:
            # Base case: compare-exchange the two elements
            a = offset
            b = offset + 1
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        
        return comparators
    
    network = bitonic_sort_rec(n, 0, True)
    return network


def _find_redundant_comparators(network, wires, time_limit):
    """Find redundant comparators by trying to remove them."""
    current = list(network)
    start_time = time.time()
    
    # Try removing comparators from end to start
    i = len(current) - 1
    while i >= 0 and time.time() - start_time < time_limit:
        candidate = current[:i] + current[i+1:]
        if _verify_network(candidate, wires):
            current = candidate
        i -= 1
    
    # Try again from the beginning since removals may have made earlier ones redundant
    i = 0
    while i < len(current) and time.time() - start_time < time_limit:
        candidate = current[:i] + current[i+1:]
        if _verify_network(candidate, wires):
            current = candidate
        else:
            i += 1
    
    return current


def _local_search(network, wires, time_limit):
    """Local search to reduce network size."""
    current = list(network)
    best = list(current)
    best_size = len(current)
    start_time = time.time()
    
    # Temperature for simulated annealing
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Try removing a random comparator
        if len(current) > 0:
            i = random.randint(0, len(current) - 1)
            candidate = current[:i] + current[i+1:]
            if _verify_network(candidate, wires):
                current = candidate
                if len(current) < best_size:
                    best = list(current)
                    best_size = len(current)
        
        # Try adding a random comparator
        a = random.randint(0, N - 2)
        b = random.randint(a + 1, N - 1)
        candidate = current + [(a, b)]
        if _verify_network(candidate, wires):
            # Adding a comparator that maintains validity means we can potentially
            # remove others, so accept it with some probability
            if random.random() < temperature:
                current = candidate
        
        # Try swapping two adjacent comparators
        if len(current) > 1:
            i = random.randint(0, len(current) - 2)
            candidate = current[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify_network(candidate, wires):
                current = candidate
        
        temperature *= cooling_rate
    
    return best


def _construct():
    """Construct an initial sorting network."""
    # Start with Batcher's odd-even merge sort
    network = _batcher_odd_even_merge_sort(N)
    
    # Verify it's correct
    wires = _build_wires()
    if not _verify_network(network, wires):
        # Fallback: odd-even transposition
        network = _odd_even_transposition(N)
    
    return network, wires


def construct():
    deadline = 85.0  # Safety margin: evaluator kills at 90s
    start = time.time()
    
    # Start with Batcher's odd-even merge sort
    network, wires = _construct()
    
    # Phase 1: Remove redundant comparators
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5
    if remaining > 10:
        network = _find_redundant_comparators(network, wires, remaining * 0.4)
    
    # Phase 2: Local search with simulated annealing
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5
    if remaining > 10:
        network = _local_search(network, wires, remaining * 0.5)
    
    # Final verification
    if not _verify_network(network, wires):
        # Fallback to original Batcher
        network = _batcher_odd_even_merge_sort(N)
    
    return network

# EVOLVE-BLOCK-END
