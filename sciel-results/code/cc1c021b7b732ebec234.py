"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin

    # Pippenger 58-comparator network for 16 wires
    network = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (1, 2), (5, 6), (9, 10), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (2, 4), (3, 5), (6, 8), (7, 9), (10, 12), (11, 13),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]

    def _verify_network_fast(comparators, n):
        """Fast verification using NumPy bit-parallelism over all 2^n inputs."""
        # Create array of all 2^n bitmasks
        masks = np.arange(1 << n, dtype=np.uint32)
        # Extract bits: wires[k] for each mask
        wires = np.zeros((1 << n, n), dtype=np.uint8)
        for k in range(n):
            wires[:, k] = (masks >> k) & 1
        
        for (i, j) in comparators:
            # Compare-exchange: min to i, max to j
            # For each input, if wires[:,i] > wires[:,j], swap
            mask_swap = wires[:, i] > wires[:, j]
            wires[mask_swap, i] = 1 - wires[mask_swap, i]
            wires[mask_swap, j] = 1 - wires[mask_swap, j]
        
        # Check sorted: wires[:,k] <= wires[:,k+1] for all k
        for k in range(n - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
        return True

    def _remove_redundant(comparators, n, deadline):
        """Remove redundant comparators one by one."""
        result = list(comparators)
        changed = True
        while changed:
            changed = False
            for idx in range(len(result) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = result[:idx] + result[idx+1:]
                if _verify_network_fast(candidate, n):
                    result = candidate
                    changed = True
            if time.time() > deadline:
                break
        return result

    def _try_swap_order(comparators, n, deadline):
        """Try swapping adjacent comparators that don't interfere."""
        result = list(comparators)
        improved = True
        while improved:
            improved = False
            for idx in range(len(result) - 1):
                if time.time() > deadline:
                    break
                i1, j1 = result[idx]
                i2, j2 = result[idx + 1]
                # Only swap if they don't share a wire (independent)
                if j1 <= i2 or j2 <= i1:
                    candidate = result[:idx] + [result[idx+1], result[idx]] + result[idx+2:]
                    if _verify_network_fast(candidate, n):
                        result = candidate
                        improved = True
                        break  # Restart scan after a swap
        return result

    # Verify initial network
    if not _verify_network_fast(network, N):
        # Fallback to Batcher's Odd-Even Merge Sort
        def _batcher_odd_even_merge_sort(n):
            if n == 1:
                return []
            half = n // 2
            left = _batcher_odd_even_merge_sort(half)
            right = [(i + half, j + half) for (i, j) in left]
            merge = _odd_even_merge(half)
            return left + right + merge

        def _odd_even_merge(n):
            if n == 1:
                return [(0, 1)]
            even_merge = _odd_even_merge(n // 2)
            odd_merge = _odd_even_merge(n // 2)
            even_translated = [(2*i, 2*j) for (i, j) in even_merge]
            odd_translated = [(2*i + 1, 2*j + 1) for (i, j) in odd_merge]
            final_merge = [(2*i + 1, 2*i + 2) for i in range(n - 1)]
            return even_translated + odd_translated + final_merge

        network = _batcher_odd_even_merge_sort(N)

    # Phase 1: Remove redundant comparators
    network = _remove_redundant(network, N, deadline)

    # Phase 2: Try to improve by swapping independent adjacent comparators
    if time.time() < deadline:
        network = _try_swap_order(network, N, deadline)

    # Phase 3: Another round of redundancy removal after swaps
    if time.time() < deadline:
        network = _remove_redundant(network, N, deadline)

    return network
# EVOLVE-BLOCK-END

