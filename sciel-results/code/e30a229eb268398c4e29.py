"""Direct construction of a size-efficient sorting network on 18 wires using Simulated Annealing."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin (90s limit, leave buffer)

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    # Precompute sortedness check: for each pair (k, k+1), the mask where wire k has 1 and wire k+1 has 0
    # If any of these masks is non-zero after sorting, the network is invalid
    sortedness_masks = [None] * (N - 1)
    for k in range(N - 1):
        # Mask where initial wire k is 1 and initial wire k+1 is 0
        # This is (initial_wires[k] & ~initial_wires[k+1])
        # But we need to track this through the network... actually we can't precompute this
        # because the wires change. We need to check at the end.
        pass

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        # Use a copy of the initial wires
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        # i.e., wires[k] & ~wires[k+1] should be all zeros
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        For small networks, try all positions. For larger, sample.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            
            # Determine which positions to try
            if n <= 50:
                # Try all positions for small networks
                positions = list(range(n))
                random.shuffle(positions)
            else:
                # For larger networks, try a larger sample
                positions = random.sample(range(n), min(50, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def odd_even_merge_network():
        """Odd-even merge sort network."""
        def merge(a, b):
            """Merge two sorted networks of size a and b."""
            if a == 1 or b == 1:
                return [(0, 1)] if a + b == 2 else []
            
            # Standard odd-even merge
            result = []
            # Odd-even transposition merge
            for phase in range(a + b):
                if phase % 2 == 0:
                    for i in range(0, a + b - 1, 2):
                        if i < a or i >= a - 1:  # Simplified
                            result.append((i, i + 1))
                else:
                    for i in range(1, a + b - 1, 2):
                        result.append((i, i + 1))
            return result
        
        # Build recursively
        def build(n):
            if n == 1:
                return []
            if n == 2:
                return [(0, 1)]
            half = n // 2
            left = build(half)
            right = build(n - half)
            # Shift right comparators
            right_shifted = [(i + half, j + half) for (i, j) in right]
            # Merge
            merged = merge(half, n - half)
            return left + right_shifted + merged
        
        return build(N)

    def random_dense_network(num_comparators=200, seed=None):
        """Generate a random dense network and shuffle."""
        if seed is not None:
            random.seed(seed)
        comparators = []
        for _ in range(num_comparators):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def simulated_annealing(initial_net, t_max=1000, t_min=0.5, alpha=0.997, max_iter=20000, deadline=None, seed=42):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        random.seed(seed)
        
        iterations_since_improvement = 0
        max_iterations_without_improvement = 500
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Restart logic
            if iterations_since_improvement > max_iterations_without_improvement:
                # Reset to best and increase temperature
                current = list(best)
                current_energy = best_energy
                t = t_max * 0.8 # Restart with slightly lower max temp
                iterations_since_improvement = 0
                max_iterations_without_improvement = int(max_iterations_without_improvement * 0.8)
                
                # If max_iterations_without_improvement is too small, break
                if max_iterations_without_improvement < 50:
                    break

            r = random.random()
            candidate = list(current)
            
            # Dynamic probabilities: if stuck, insert more
            p_remove = 0.5
            p_insert = 0.15
            p_swap = 0.2
            p_replace = 0.15
            
            if iterations_since_improvement > 100:
                # Increase insert probability to escape
                p_remove = 0.4
                p_insert = 0.3
                p_swap = 0.15
                p_replace = 0.15

            if r < p_remove:
                # Remove
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif r < p_remove + p_insert:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
            elif r < p_remove + p_insert + p_swap:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            else:
                # Replace
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                candidate[idx] = (u, v)

            if not verify_network(candidate):
                continue # Invalid, reject

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                # Always accept improvement
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
                    iterations_since_improvement = 0
            else:
                # Accept with probability exp(-delta / T)
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / t):
                        current = candidate
                        current_energy = new_energy
                        iterations_since_improvement += 1
                else:
                    # Should not happen if new_energy >= current_energy
                    pass
            
            if new_energy == current_energy:
                iterations_since_improvement += 1

            t *= alpha
            if t < t_min:
                break
        
        return best

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    
    # Add more random dense starts with different densities
    # Remove odd_even_merge as it is buggy/inefficient
    for i in range(5):
        density = [120, 140, 160, 180, 200][i]
        candidates.append((f"random_{i}", random_dense_network(density, seed=100 + i * 7)))

    best_network = None
    best_count = float('inf')

    # Run SA on each candidate
    for idx, (name, net) in enumerate(candidates):
        if time.time() > deadline - 8.0:
            break
        
        # First, try to prune the initial network to get a better starting point for SA
        prune_time = min(start_time + 3.0, deadline - 10.0)
        pruned = greedy_remove(net, time_limit=prune_time)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        # Use more iterations for later candidates if time permits
        remaining_time = deadline - 8.0 - time.time()
        if remaining_time < 5.0:
            break
            
        sa_result = simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.997, 
            max_iter=15000, 
            deadline=deadline - 6.0,
            seed=42 + idx * 100
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)
            
        # Final greedy remove on the SA result
        final_pruned = greedy_remove(best_network, time_limit=deadline - 3.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        # Fallback
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        # Should not happen, but safety
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
