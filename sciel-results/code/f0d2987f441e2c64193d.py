"""Size-efficient sorting networks on 16 wires."""
import time
import numpy as np

N = 16
DEADLINE_MARGIN = 5.0


def _is_sorted_network(net):
    """Check if network sorts all inputs using zero-one principle with numpy."""
    if not net:
        return N <= 1
    
    all_inputs = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in net:
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        bit_i_set = (all_inputs & mask_i) != 0
        bit_j_set = (all_inputs & mask_j) != 0
        
        # Swap if bit_i is 1 and bit_j is 0
        swap_mask = bit_i_set & ~bit_j_set
        
        all_inputs[swap_mask] &= ~mask_i
        all_inputs[swap_mask] |= mask_j
    
    # Valid sorted outputs: numbers of form (2^N - 2^k) for k=0..N
    valid = np.zeros(1 << N, dtype=bool)
    for k in range(N + 1):
        valid[(1 << N) - (1 << k)] = True
    
    return bool(np.all(valid[all_inputs]))


def _batcher_bitonic_16():
    """Generate Batcher's bitonic sorting network for 16 inputs."""
    net = []
    
    def bitonic_merge(start, size, ascending=True):
        if size == 2:
            if ascending:
                net.append((start, start + 1))
            else:
                net.append((start + 1, start))
            return
        half = size // 2
        # Compare corresponding elements from the two halves
        for i in range(half):
            if ascending:
                net.append((start + i, start + half + i))
            else:
                net.append((start + half + i, start + i))
        # Recursively sort the two halves
        bitonic_merge(start, half, ascending)
        bitonic_merge(start + half, half, ascending)
    
    def bitonic_sort(start, size, ascending=True):
        if size == 2:
            bitonic_merge(start, size, ascending)
            return
        half = size // 2
        bitonic_sort(start, half, True)
        bitonic_sort(start + half, half, False)
        bitonic_merge(start, size, ascending)
    
    bitonic_sort(0, 16, True)
    return net


def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Start with Batcher's bitonic sort
    net = _batcher_bitonic_16()
    
    # Verify correctness
    if not _is_sorted_network(net):
        # Fallback to odd-even transposition
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(net)
    best_count = len(best_net)
    
    # Greedy removal: try to remove each comparator if the network remains valid
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx + 1:]
            if _is_sorted_network(candidate):
                net = candidate
                improved = True
                if len(net) < best_count:
                    best_count = len(net)
                    best_net = list(net)
                break
    
    return best_net
