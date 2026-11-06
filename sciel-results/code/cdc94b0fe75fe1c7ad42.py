import numpy as np
import time
import random

N = 16
ALL_MASKS = np.arange(1 << N, dtype=np.int32)
SORTED_ARRAY = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    SORTED_ARRAY[(1 << k) - 1] = True
UNSORTED_ARRAY = ~SORTED_ARRAY

def _merge(k):
    if k == 1:
        return [(0, 1)]
    k_half = k // 2
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    return even_comps + odd_comps + final_comps

def _batcher(n):
    if n == 1:
        return []
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    return left + right + merge

def _bitonic(n):
    """Generate Batcher's Bitonic Sort network for n wires."""
    if n <= 1:
        return []
    
    def _bitonic_stage(k, direction=1, offset=0):
        """Generate comparators for bitonic stage of size k."""
        comps = []
        if k == 1:
            return comps
        half = k // 2
        # Recursively sort first half ascending, second half descending
        left = _bitonic_stage(half, 1, offset)
        right = _bitonic_stage(half, -1, offset + half)
        comps.extend(left)
        comps.extend(right)
        # Merge phase
        for i in range(half):
            for j in range(half - i - 1, -1, -1):
                a = offset + i + j
                b = offset + i + j + half
                if direction == 1:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
        return comps
    
    def _bitonic_sort(k, direction=1, offset=0):
        """Full bitonic sort for a block of size k."""
        if k <= 1:
            return []
        if k == 2:
            if direction == 1:
                return [(offset, offset + 1)]
            else:
                return [(offset + 1, offset)]
        
        half = k // 2
        # Create bitonic sequence
        left = _bitonic_sort(half, 1, offset)
        right = _bitonic_sort(half, -1, offset + half)
        comps = left + right
        
        # Merge the bitonic sequence
        def _merge_bitonic(start, length, direction):
            if length <= 1:
                return []
            if length == 2:
                a = start
                b = start + 1
                if direction == 1:
                    return [(a, b)]
                else:
                    return [(b, a)]
            
            half_len = length // 2
            comps = []
            # Recursively merge halves
            comps.extend(_merge_bitonic(start, half_len, direction))
            comps.extend(_merge_bitonic(start + half_len, half_len, direction))
            # Cross comparisons
            for i in range(half_len):
                for j in range(half_len - i - 1, -1, -1):
                    a = start + i + j
                    b = start + i + j + half_len
                    if direction == 1:
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
            return comps
        
        comps.extend(_merge_bitonic(offset, k, direction))
        return comps
    
    return _bitonic_sort(n, 1, 0)

def _simulate_network(network, masks):
    m = masks.copy()
    for i, j in network:
        bit_i = (m >> i) & 1
        bit_j = (m >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            m = np.where(swap_mask, (m & ~(1 << i)) | (1 << j), m)
    return m

def _full_check_fast(network):
    results = _simulate_network(network, ALL_MASKS)
    return np.all(SORTED_ARRAY[results])

def _count_unsorted(network):
    results = _simulate_network(network, ALL_MASKS)
    return int(np.sum(~SORTED_ARRAY[results]))

def _try_remove_single(network, idx):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    new_network = network[:idx] + network[idx+1:]
    if _full_check_fast(new_network):
        return True, new_network
    return False, None

def _try_remove_pair(network, idx1, idx2):
    """Try removing two comparators. Returns (success, new_network)."""
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    new_network = network[:idx1] + network[idx1+1:idx2] + network[idx2+1:]
    if _full_check_fast(new_network):
        return True, new_network
    return False, None

def _prune_single(network, deadline, max_passes=1000):
    """Aggressively remove redundant comparators one at a time."""
    network = list(network)
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        removed = False
        n_comps = len(network)
        if n_comps == 0:
            break
        # Try from both ends and middle
        order = list(range(n_comps - 1, -1, -1))
        mid = n_comps // 2
        order = order[mid:] + order[:mid][::-1]
        for idx in order:
            if time.time() > deadline:
                break
            success, new_network = _try_remove_single(network, idx)
            if success:
                network = new_network
                removed = True
                break
        if not removed:
            break
    return network

def _prune_pairs(network, deadline, max_passes=500):
    """Try removing pairs of comparators."""
    network = list(network)
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        removed = False
        n_comps = len(network)
        if n_comps < 2:
            break
        # Try adjacent pairs first, then some random pairs
        rng = random.Random(hash(tuple(network)) % (2**31))
        tried = set()
        # Adjacent pairs
        for i in range(n_comps - 1):
            if time.time() > deadline:
                break
            if (i, i+1) in tried:
                continue
            tried.add((i, i+1))
            success, new_network = _try_remove_pair(network, i, i+1)
            if success:
                network = new_network
                removed = True
                break
        if removed:
            continue
        # Random pairs
        for _ in range(min(50, n_comps * (n_comps - 1) // 2)):
            if time.time() > deadline:
                break
            i = rng.randint(0, n_comps - 1)
            j = rng.randint(0, n_comps - 1)
            if i == j or (i, j) in tried or (j, i) in tried:
                continue
            tried.add((min(i,j), max(i,j)))
            success, new_network = _try_remove_pair(network, i, j)
            if success:
                network = new_network
                removed = True
                break
        if not removed:
            break
    return network

def _local_search_replace(network, deadline, max_iters=200):
    """Try replacing comparators to enable further removals."""
    network = list(network)
    rng = random.Random(42)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    best = list(network)
    best_count = len(best)
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        n_comps = len(best)
        if n_comps == 0:
            break
        
        improved = False
        # Try replacing a random comparator
        for _ in range(min(15, n_comps)):
            if time.time() > deadline:
                break
            idx = rng.randint(0, n_comps - 1)
            new_comp = rng.choice(all_pairs)
            if new_comp == best[idx]:
                continue
            new_network = best[:idx] + [new_comp] + best[idx+1:]
            if _full_check_fast(new_network):
                # Try to remove something from this new network
                pruned = _prune_single(new_network, deadline - 2, max_passes=20)
                if len(pruned) < best_count:
                    best = pruned
                    best_count = len(pruned)
                    improved = True
                    break
        
        if not improved:
            # Try moving a comparator
            if n_comps >= 2:
                for _ in range(min(10, n_comps)):
                    if time.time() > deadline:
                        break
                    i = rng.randint(0, n_comps - 1)
                    j = rng.randint(0, n_comps - 1)
                    if i == j:
                        continue
                    new_network = list(best)
                    new_network[i], new_network[j] = new_network[j], new_network[i]
                    if _full_check_fast(new_network):
                        pruned = _prune_single(new_network, deadline - 2, max_passes=20)
                        if len(pruned) < best_count:
                            best = pruned
                            best_count = len(pruned)
                            improved = True
                            break
        
        if not improved and _iter > 50:
            break
    
    return best

def _build_from_bitonic(deadline):
    """Build a network starting from bitonic sort topology."""
    network = _bitonic(N)
    # Prune singles
    network = _prune_single(network, deadline - 10)
    # Prune pairs
    network = _prune_pairs(network, deadline - 5)
    # Local search
    network = _local_search_replace(network, deadline - 2)
    # Final prune
    network = _prune_single(network, deadline)
    return network

def _build_from_batcher(deadline):
    """Build a network starting from Batcher's odd-even merge topology."""
    network = _batcher(N)
    network = _prune_single(network, deadline - 10)
    network = _prune_pairs(network, deadline - 5)
    network = _local_search_replace(network, deadline - 2)
    network = _prune_single(network, deadline)
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_network = None
    best_count = float('inf')
    
    # Strategy 1: Start from bitonic sort topology
    if time.time() < deadline - 30:
        net1 = _build_from_bitonic(deadline - 25)
        if _full_check_fast(net1):
            best_network = net1
            best_count = len(net1)
    
    # Strategy 2: Start from Batcher's odd-even merge topology
    if time.time() < deadline - 30:
        net2 = _build_from_batcher(deadline - 25)
        if _full_check_fast(net2) and len(net2) < best_count:
            best_network = net2
            best_count = len(net2)
    
    # Strategy 3: Hybrid - take the better of the two, then cross-prune
    if best_network is not None and time.time() < deadline - 10:
        # Try to further optimize the best network
        final_net = _prune_pairs(best_network, deadline - 5)
        final_net = _local_search_replace(final_net, deadline - 2)
        final_net = _prune_single(final_net, deadline)
        if _full_check_fast(final_net) and len(final_net) < best_count:
            best_network = final_net
            best_count = len(final_net)
    
    # Fallback: if nothing worked, use Batcher directly
    if best_network is None:
        best_network = _batcher(N)
        best_network = _prune_single(best_network, deadline)
    
    # Final verification
    if not _full_check_fast(best_network):
        best_network = _batcher(N)
    
    return best_network
