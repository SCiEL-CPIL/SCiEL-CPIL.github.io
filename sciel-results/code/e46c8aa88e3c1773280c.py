"""Sorting network for 18 wires using optimized search."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

# Precompute bit-parallel verification data
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)  # shape (N, 2^N)

def _verify_network_bitparallel(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    wires = _WIRE_BITS.copy()  # shape (N, 2^N)
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _verify_network(net):
    """Verify that a network sorts all 2^N Boolean inputs."""
    return _verify_network_bitparallel(net)

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _bitonic_merge(n, start, reverse=False):
    """
    Generate bitonic merge network for n wires starting at position start.
    If reverse, the bitonic sequence is in reverse order.
    """
    if n <= 1:
        return []
    
    comparators = []
    # Compare elements i and i + n/2 apart
    half = n // 2
    for i in range(half):
        a = start + i
        b = start + i + half
        if reverse:
            a, b = b, a
        if a > b:
            a, b = b, a
        comparators.append((a, b))
    
    # Recursively merge halves
    left = _bitonic_merge(half, start, reverse=reverse)
    right = _bitonic_merge(half, start + half, reverse=not reverse)
    comparators.extend(left)
    comparators.extend(right)
    
    return comparators

def _bitonic_sort(n, start=0):
    """
    Generate a bitonic sort network for n wires starting at position start.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(start, start + 1)]
    
    half = n // 2
    other = n - half
    
    # Sort first half ascending, second half descending
    left = _bitonic_sort(half, start)
    # For the right half, we need to sort it in reverse order
    # Bitonic sort in reverse: sort ascending then reverse
    right_raw = _bitonic_sort(other, start + half)
    # To get descending, we can use a trick: sort ascending then flip
    # Actually, let's just generate a descending sort by reversing the comparator order
    # and swapping the roles
    # Simpler: generate ascending sort for right half, then we'll handle in merge
    
    # Actually, the standard bitonic sort:
    # 1. Sort first half ascending
    # 2. Sort second half descending
    # 3. Merge
    
    # For descending sort of right half, we can generate ascending sort
    # but the comparators will naturally work if we just generate them
    # Let me use a different approach: generate the full bitonic sort recursively
    
    comparators = left + right_raw
    # Now merge the two halves (one ascending, one descending)
    # The bitonic merge assumes the first half is ascending and second is descending
    # So we need the right half to be in descending order
    # Let's just generate a proper bitonic sort
    
    return comparators

def _bitonic_sort_proper(n, start=0, reverse=False):
    """
    Generate a proper bitonic sort network.
    """
    if n <= 1:
        return []
    if n == 2:
        a, b = start, start + 1
        if reverse:
            return [(a, b)]  # min goes to a, max goes to b - this is ascending
        # For reverse, we want descending: max to a, min to b
        # But our comparator puts min to first, max to second
        # So for descending, we swap: (b, a) -> min to b, max to a
        # But we need i < j, so we just use (a, b) and the logic handles it
        return [(a, b)]
    
    half = n // 2
    other = n - half
    
    # Sort first half in one direction, second half in opposite
    if not reverse:
        left = _bitonic_sort_proper(half, start, reverse=False)
        right = _bitonic_sort_proper(other, start + half, reverse=True)
    else:
        left = _bitonic_sort_proper(half, start, reverse=True)
        right = _bitonic_sort_proper(other, start + half, reverse=False)
    
    comparators = left + right
    
    # Now merge
    # Bitonic merge: compare i with i + n/2, then recursively merge
    # The merge works on the combined n elements
    comparators.extend(_bitonic_merge(n, start))
    
    return comparators

def _build_bitonic_network():
    """Build a bitonic sort network for N wires."""
    # Bitonic sort requires power of 2, so we pad to 32
    # But we only use 18 wires
    # Let's use a simpler approach: odd-even merge sort
    
    # Actually, let's just use a known good network
    # For 18 wires, we can use a combination of smaller networks
    
    # Let's use the AKS-style approach or just a simple recursive merge sort
    return _odd_even_merge_sort(N)

def _odd_even_merge_sort(n):
    """
    Generate an odd-even merge sort network for n wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    other = n - half
    
    # Sort first half
    left_net = _odd_even_merge_sort(half)
    # Sort second half (shifted)
    right_net_raw = _odd_even_merge_sort(other)
    right_net = [(i + half, j + half) for i, j in right_net_raw]
    
    # Merge the two halves using odd-even merge
    merge_net = _odd_even_merge(half, other)
    
    return left_net + right_net + merge_net

def _odd_even_merge(m, n):
    """
    Odd-even merge network to merge two sorted sequences of lengths m and n.
    Returns a list of comparators (i, j) with 0 <= i < j < m+n.
    """
    if m == 0 or n == 0:
        return []
    if m == 1 and n == 1:
        return [(0, 1)]
    
    comparators = []
    total = m + n
    
    # Step 1: Compare corresponding elements
    for i in range(min(m, n)):
        a, b = i, m + i
        if a > b:
            a, b = b, a
        comparators.append((a, b))
    
    # Step 2: Split into even and odd subsequences
    even_wires = list(range(0, total, 2))
    odd_wires = list(range(1, total, 2))
    
    # Determine how many elements from each original sequence go to even/odd
    m_even = (m + 1) // 2
    m_odd = m // 2
    
    if m % 2 == 0:
        n_even = (n + 1) // 2
        n_odd = n // 2
    else:
        n_even = n // 2
        n_odd = (n + 1) // 2
    
    # Recursively merge
    even_comps = _odd_even_merge(m_even, n_even)
    # Remap even comparators to actual wire positions
    for a, b in even_comps:
        actual_a = even_wires[a]
        actual_b = even_wires[b]
        if actual_a > actual_b:
            actual_a, actual_b = actual_b, actual_a
        comparators.append((actual_a, actual_b))
    
    odd_comps = _odd_even_merge(m_odd, n_odd)
    for a, b in odd_comps:
        actual_a = odd_wires[a]
        actual_b = odd_wires[b]
        if actual_a > actual_b:
            actual_a, actual_b = actual_b, actual_a
        comparators.append((actual_a, actual_b))
    
    return comparators

def _local_search(net, max_time=50.0):
    """
    Local search to improve the network.
    Uses greedy deletion, then targeted replacement.
    """
    start_time = time.time()
    best_net = list(net)
    best_len = len(best_net)
    
    def _elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy deletion
    while _elapsed() < max_time * 0.3:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time * 0.3:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                best_len = len(best_net)
                improved = True
        if not improved:
            break
    
    # Phase 2: Try replacing comparators to enable further deletions
    while _elapsed() < max_time * 0.6:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time * 0.6:
                break
            i, j = best_net[idx]
            # Try nearby replacements
            for a in range(max(0, i-3), min(N, i+4)):
                for b in range(a+1, min(N, j+4)):
                    if (a, b) == (i, j):
                        continue
                    candidate = list(best_net)
                    candidate[idx] = (a, b)
                    if _verify_network(candidate):
                        best_net = candidate
                        best_len = len(best_net)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # Try deleting again
                for idx2 in range(len(best_net)):
                    if _elapsed() >= max_time * 0.6:
                        break
                    candidate = best_net[:idx2] + best_net[idx2+1:]
                    if _verify_network(candidate):
                        best_net = candidate
                        best_len = len(best_net)
                        break
        if not improved:
            break
    
    # Phase 3: More aggressive search
    while _elapsed() < max_time:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time:
                break
            i, j = best_net[idx]
            # Try all possible replacements
            for a in range(N):
                if _elapsed() >= max_time:
                    break
                for b in range(a+1, N):
                    if (a, b) == (i, j):
                        continue
                    candidate = list(best_net)
                    candidate[idx] = (a, b)
                    if _verify_network(candidate):
                        best_net = candidate
                        best_len = len(best_net)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # Try deleting again
                for idx2 in range(len(best_net)):
                    if _elapsed() >= max_time:
                        break
                    candidate = best_net[:idx2] + best_net[idx2+1:]
                    if _verify_network(candidate):
                        best_net = candidate
                        best_len = len(best_net)
                        break
        if not improved:
            break
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 55.0  # Leave 5s margin for safety
    
    # Start with odd-even merge sort network
    net = _odd_even_merge_sort(N)
    
    # Normalize pairs
    normalized = []
    for i, j in net:
        if i > j:
            i, j = j, i
        if 0 <= i < j < N:
            normalized.append((i, j))
    net = normalized
    
    # Verify it's correct
    if not _verify_network(net):
        # Fallback: use odd-even transposition (guaranteed correct)
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
    
    # Local search to improve
    remaining = deadline - time.time()
    if remaining > 3.0:
        net = _local_search(net, max_time=min(remaining - 3.0, 50.0))
    
    # Final safety check
    final_net = []
    for i, j in net:
        if 0 <= i < j < N:
            final_net.append((i, j))
    
    # Final verification
    if not _verify_network(final_net):
        # Absolute fallback
        final_net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    final_net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    final_net.append((i, i + 1))
    
    return final_net
# EVOLVE-BLOCK-END
