"""Size-efficient sorting networks on 18 wires.

Strategy:
1. Start from the odd-even transposition network (153 comparators) as a valid baseline.
2. Use a local search / simulated annealing approach to remove redundant comparators.
3. For each candidate comparator removal, verify correctness on a representative set
   of Boolean inputs (all 2^18 = 262144 inputs, but we can use the 0-1 principle
   which is sufficient for sorting networks).
4. To speed up verification, we use vectorized NumPy operations to test many inputs
   at once.
5. We also try a divide-and-conquer construction: sort 9+9 then merge, which can
   give fewer comparators than odd-even transposition.

The key insight is that the 0-1 principle states that a comparator network sorts
all inputs if and only if it sorts all 0-1 (Boolean) inputs. So we need to check
all 2^18 = 262144 Boolean inputs. We can do this efficiently with NumPy by
processing them in batches.

We'll use a greedy removal approach: try removing each comparator and check if
the network still sorts all Boolean inputs. If it does, keep it removed.

To make this feasible within the time budget, we'll:
- Precompute the effect of the network on all Boolean inputs using matrix operations
- Use a smarter verification that checks only a subset of critical inputs first
- Limit the search to a reasonable number of iterations

Actually, checking all 262144 inputs for each removal is too slow if done naively.
Let's think differently.

A better approach: Use the known optimal or near-optimal networks from the literature.
The best known for 18 wires is 77 comparators. Let's try to construct a good network
using a divide-and-conquer approach:
- Sort the first 9 wires
- Sort the last 9 wires  
- Merge the two sorted halves

For sorting 9 wires, we can use a known small network. The optimal for 9 wires is
20 comparators (known result). For merging two sorted 9-wire sequences, we can use
an odd-even merge network.

Actually, let's use a simpler but effective approach:
1. Use bitonic sort for 16 wires (80 comparators) and handle the remaining 2 wires.
2. Or use a known good network and apply local search.

Let me implement a practical approach:
1. Generate a reasonable initial network (odd-even transposition or divide-and-conquer)
2. Use local search to remove redundant comparators
3. Verify correctness using the 0-1 principle with efficient NumPy-based checking

For efficiency, I'll verify correctness by simulating the network on all 2^18 Boolean
inputs, but I'll do it in a vectorized way.
"""

import time
import numpy as np

N = 18
DEADLINE = 85.0  # seconds, leaving 5s safety margin for the 90s limit
_start_time = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - _start_time)

def _apply_network(comps, inputs):
    """Apply comparator network to a set of inputs.
    inputs: shape (num_inputs, N) array of values
    comps: list of (i, j) comparators
    Returns sorted outputs.
    """
    x = inputs.copy()
    for i, j in comps:
        # min goes to i, max goes to j
        tmp = np.minimum(x[:, i], x[:, j])
        x[:, i] = tmp
        x[:, j] = np.maximum(x[:, i], x[:, j])
    return x

def _is_sorting_network(comps, num_checks=None):
    """Check if the network sorts all Boolean inputs using the 0-1 principle.
    For full verification, we need to check all 2^N inputs.
    For speed, we can check a random subset, but for correctness we need all.
    """
    if num_checks is None:
        num_checks = 2 ** N
    
    # Generate all Boolean inputs
    # Use a batched approach for memory efficiency
    batch_size = min(num_checks, 65536)
    
    for start in range(0, num_checks, batch_size):
        end = min(start + batch_size, num_checks)
        count = end - start
        
        # Generate batch of inputs
        # Each input is an N-bit number
        indices = np.arange(start, end)
        # Convert to binary
        inputs = np.zeros((count, N), dtype=np.int8)
        for bit in range(N):
            inputs[:, bit] = (indices >> bit) & 1
        
        # Apply network
        x = inputs.copy()
        for i, j in comps:
            tmp = np.minimum(x[:, i], x[:, j])
            x[:, i] = tmp
            x[:, j] = np.maximum(x[:, i], x[:, j])
        
        # Check if sorted: each row should be non-decreasing
        # For 0-1 inputs, sorted means all 0s come before all 1s
        # So the sorted output should have the form: 0...0 1...1
        # Check: for each row, find the position where it transitions from 0 to 1
        # All rows with the same number of 1s should be identical
        
        # A faster check: the output should be sorted, i.e., x[:, k] <= x[:, k+1] for all k
        if not np.all(x[:, :-1] <= x[:, 1:]):
            return False
    
    return True

def _odd_even_transposition(n):
    """Return odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _bitonic_sort_network(n):
    """Return bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Recursively sort first half ascending, second half descending
    # Then merge
    
    comps = []
    # Sort first half
    comps.extend(_bitonic_sort_network(half))
    # Sort second half (we'll handle direction in the merge)
    comps.extend([(i + half, j + half) for i, j in _bitonic_sort_network(half)])
    
    # Bitonic merge
    comps.extend(_bitonic_merge(n))
    
    return comps

def _bitonic_merge(n):
    """Return bitonic merge network for n wires (n power of 2)."""
    if n == 1:
        return []
    
    comps = []
    k = n // 2
    # Compare elements k apart
    for i in range(k):
        comps.append((i, i + k))
    
    # Recursively merge the two halves
    comps.extend(_bitonic_merge(k))
    comps.extend([(i + k, j + k) for i, j in _bitonic_merge(k)])
    
    return comps

def _divide_and_conquer(n):
    """Construct a sorting network using divide and conquer:
    Sort first half, sort second half, then merge.
    """
    if n <= 2:
        return [(0, 1)] if n == 2 else []
    
    half = n // 2
    comps = []
    
    # Sort first half
    comps.extend(_divide_and_conquer(half))
    
    # Sort second half
    comps.extend([(i + half, j + half) for i, j in _divide_and_conquer(n - half)])
    
    # Merge the two sorted halves
    # Use a simple merge: compare corresponding elements and fix
    # This is not optimal but gives a valid network
    # For a proper merge, we need a merge network
    # Let's use a simple approach: apply odd-even transposition on the combined wires
    # but only the ones that need it
    
    # Actually, let's use a proper merge network
    comps.extend(_merge_networks(half, n - half))
    
    return comps

def _merge_networks(m, n):
    """Create a merge network for two sorted sequences of length m and n.
    This is a simplified version that may not be optimal.
    """
    # For simplicity, use odd-even transposition on the combined wires
    # This is not minimal but correct
    total = m + n
    return _odd_even_transposition(total)

def _local_search(comps, max_iter=200):
    """Try to remove redundant comparators using local search."""
    comps = list(comps)
    n_comps = len(comps)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        # Try removing each comparator
        for idx in range(len(comps)):
            if _time_remaining() < 5.0:
                break
            
            # Try removing this comparator
            trial = comps[:idx] + comps[idx+1:]
            
            # Quick check: verify on a small random sample first
            if not _quick_check(trial):
                continue
            
            # Full check
            if _is_sorting_network(trial):
                comps = trial
                improved = True
                break
        
        if not improved:
            break
    
    return comps

def _quick_check(comps, num_samples=1000):
    """Quick check on random Boolean inputs."""
    rng = np.random.default_rng(42)
    inputs = rng.integers(0, 2, size=(num_samples, N), dtype=np.int8)
    
    x = inputs.copy()
    for i, j in comps:
        tmp = np.minimum(x[:, i], x[:, j])
        x[:, i] = tmp
        x[:, j] = np.maximum(x[:, i], x[:, j])
    
    if not np.all(x[:, :-1] <= x[:, 1:]):
        return False
    return True

def construct():
    """Construct a sorting network for 18 wires."""
    # Start with odd-even transposition as baseline
    comps = _odd_even_transposition(N)
    
    # Try local search to remove redundant comparators
    # But first, let's try a divide-and-conquer approach which might give a better starting point
    
    # Approach 1: Odd-even transposition + local search
    if _time_remaining() > 10.0:
        comps_ls = _local_search(comps, max_iter=50)
        if len(comps_ls) < len(comps):
            comps = comps_ls
    
    # Approach 2: Try divide-and-conquer construction
    if _time_remaining() > 10.0:
        # Sort 9+9 then merge
        comps_dac = []
        # Sort first 9
        comps_dac.extend(_odd_even_transposition(9))
        # Sort last 9
        comps_dac.extend([(i + 9, j + 9) for i, j in _odd_even_transposition(9)])
        # Merge: use odd-even transposition on all 18 (this is a valid merge)
        # But we can do better
        # For now, just add comparators that connect the two halves
        # A simple merge: compare i with i+9 for i in 0..8, then fix up
        for i in range(9):
            comps_dac.append((i, i + 9))
        # Then do a few rounds of fixing
        for phase in range(18):
            for i in range(phase % 2, 17, 2):
                comps_dac.append((i, i + 1))
        
        # Check if this is valid and better
        if _quick_check(comps_dac):
            if _is_sorting_network(comps_dac):
                if len(comps_dac) < len(comps):
                    comps = comps_dac
                    # Try local search on this
                    if _time_remaining() > 10.0:
                        comps_ls2 = _local_search(comps, max_iter=50)
                        if len(comps_ls2) < len(comps):
                            comps = comps_ls2
    
    # Final local search
    if _time_remaining() > 5.0:
        comps_final = _local_search(comps, max_iter=30)
        if len(comps_final) < len(comps):
            comps = comps_final
    
    # Ensure we have a valid network
    if not _is_sorting_network(comps):
        # Fallback to odd-even transposition
        comps = _odd_even_transposition(N)
    
    return comps
