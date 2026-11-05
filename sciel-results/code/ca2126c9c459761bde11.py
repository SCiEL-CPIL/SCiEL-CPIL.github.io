"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 18

# EVOLVE-BLOCK-START
def _apply_network(net, bits):
    """Apply a sorting network to a list of bits. Returns sorted list."""
    vals = list(bits)
    for i, j in net:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _is_sorted(vals):
    for k in range(len(vals) - 1):
        if vals[k] > vals[k+1]:
            return False
    return True

def _verify_network(net):
    """Verify that a network sorts all 2^N Boolean inputs."""
    for mask in range(1 << N):
        bits = [(mask >> i) & 1 for i in range(N)]
        if not _is_sorted(_apply_network(net, bits)):
            return False
    return True

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _merge_sort_network(n):
    """Classic merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _merge_sort_network(half)
    right = _merge_sort_network(n - half)
    # Shift right network
    right_shifted = [(i + half, j + half) for i, j in right]
    # Merge network
    merge = _merge_network(half, n - half)
    return left + right_shifted + merge

def _merge_network(m, n):
    """Merge network for two sorted inputs of sizes m and n."""
    if m == 0:
        return []
    if n == 0:
        return []
    if m == 1 and n == 1:
        return [(0, 1)]
    if m == 1:
        # Insert one element into sorted list of size n
        # This is an odd-even merge: compare (0, n-1)
        return [(0, n - 1)]
    if n == 1:
        return [(0, m - 1)]
    # Odd-even merge
    # Split into odd and even positions
    # For simplicity, use a basic merge: compare (i, m+i) for i in range(min(m,n))
    # Then recursively merge the results
    # Actually, let's use the standard odd-even merge
    # But for simplicity and correctness, let's just do a simple merge
    # Compare (i, m+i) for i in range(min(m,n))
    # Then the two halves are: positions 0..m-1 and m..m+n-1
    # After first round, positions 0..min(m,n)-1 have min of pairs
    # positions min(m,n)..m+n-1 have max of pairs (but this isn't quite right)
    
    # Let's use a simpler but correct merge:
    # We'll just do all comparisons needed
    # Actually, for small sizes, let's use a known correct approach
    # For m=n=2: compare (0,2), (1,3), then (0,1), (2,3), then (1,2)
    # Let's just use the recursive odd-even merge properly
    
    # Odd-even merge:
    # 1. Compare (i, m+i) for i in range(min(m,n))
    # 2. Recursively merge the even-indexed results and odd-indexed results
    
    # This is getting complex. Let's use a simpler approach:
    # Just generate a valid merge by comparing all cross pairs and then sorting
    # Actually, let me use a known correct merge network generator
    
    # For simplicity, let's just return a network that does a full sort of the combined input
    # This won't be optimal but will be correct
    all_pairs = []
    for i in range(m + n):
        for j in range(i + 1, m + n):
            all_pairs.append((i, j))
    # But this is too many. Let's use a better approach.
    
    # Actually, let's just use the fact that we can always add more comparators
    # to make it correct. For the merge, we'll use a simple approach:
    # Compare (i, m+i) for i in range(min(m,n))
    # Then recursively merge the two resulting sub-arrays
    
    # For now, let's just return a simple but potentially non-optimal merge
    result = []
    # First round: compare corresponding elements
    for i in range(min(m, n)):
        result.append((i, m + i))
    
    # Now we have two sorted sub-arrays:
    # positions 0, 2, 4, ... (even indices from first round)
    # positions 1, 3, 5, ... (odd indices from first round)
    # Wait, this isn't quite right either.
    
    # Let me just use a brute force merge for small sizes
    if m + n <= 8:
        # Generate all possible merge networks and pick a valid one
        # Actually, let's just use a known correct merge
        # For m=1, n=k: compare (0, k-1)
        # For m=k, n=1: compare (0, k-1)
        if m == 1:
            return [(0, n - 1)]
        if n == 1:
            return [(0, m - 1)]
    
    # For larger merges, let's use a simple approach:
    # Just do a full insertion sort style merge
    # This will be correct but not optimal
    for i in range(1, m + n):
        for j in range(i):
            result.append((j, i))
    
    return result

def _bitonic_sort_network(n):
    """Bitonic sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Sort first half ascending, second half descending
    left = _bitonic_sort_network(half)
    right = _bitonic_sort_network(n - half)
    # Shift right
    right_shifted = [(i + half, j + half) for i, j in right]
    # Now we need to make the second half descending
    # Reverse the comparators in the right network
    right_reversed = [(j, i) for i, j in right]
    right_reversed_shifted = [(i + half, j + half) for i, j in right_reversed]
    # Merge
    merge = _bitonic_merge(n)
    return left + right_reversed_shifted + merge

def _bitonic_merge(n):
    """Bitonic merge network."""
    if n <= 1:
        return []
    half = n // 2
    # Compare (i, i + n/2) for i in range(n/2)
    result = [(i, i + half) for i in range(half)]
    # Recursively merge the two halves
    left = _bitonic_merge(half)
    right = _bitonic_merge(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    return result + left + right_shifted

def _build_initial_network():
    """Build an initial sorting network using bitonic sort."""
    return _bitonic_sort_network(N)

def _local_search(net, max_time=50.0):
    """Local search to improve the network."""
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try removing each comparator
        for idx in range(len(best_net)):
            candidate = best_net[:idx] + best_net[idx+1:]
            if len(candidate) < best_score and _verify_network(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        if not improved:
            # Try replacing one comparator with another
            for idx in range(len(best_net)):
                i, j = best_net[idx]
                # Try swapping with other possible comparators
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (i, j):
                            continue
                        candidate = list(best_net)
                        candidate[idx] = (a, b)
                        if len(candidate) == best_score and _verify_network(candidate):
                            best_net = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Try adding a comparator
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) in best_net:
                            continue
                        candidate = list(best_net)
                        candidate.append((a, b))
                        if _verify_network(candidate):
                            best_net = candidate
                            best_score = len(best_net)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    # Start with a bitonic sort network
    net = _build_initial_network()
    
    # Verify it's correct
    if not _verify_network(net):
        # Fallback to odd-even transposition
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Local search to improve
    if time.time() - start_time < deadline - 5.0:
        net = _local_search(net, max_time=min(30.0, deadline - time.time() - 5.0))
    
    return net
# EVOLVE-BLOCK-END
