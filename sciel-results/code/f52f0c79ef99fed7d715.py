"""
Size-efficient sorting network on 18 wires.
Strategy: Start from a known-good network (odd-even transposition), then use
simulated annealing with bit-parallel verification to minimize comparator count.
The search allows adding, removing, and swapping comparators, with acceptance
based on energy = count + penalty*(violations). This escapes local minima that
greedy removal gets stuck in.
"""

N = 18
import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # ---- Pre-compute bit masks for verification ----
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

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def count_violations(comparators):
        """
        Count the number of wire pairs (k, k+1) where the sortedness condition
        is violated. Returns 0 if sorted.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        violations = 0
        for k in range(N - 1):
            # Count how many of the 4096 words have a violation
            bad = wires[k] & ~wires[k + 1]
            # Count non-zero words (each word represents 64 inputs)
            # Actually we need to count individual bits, but for penalty purposes
            # counting words with any violation is a good proxy
            violations += int(np.any(bad))
        return violations

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

    def odd_even_merge_sort_network(n):
        """Generate an odd-even merge sort network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into halves
        half = n // 2
        if n % 2 == 1:
            # For odd n, this is tricky; fall back to transposition for simplicity
            return odd_even_transposition()
        
        left = odd_even_merge_sort_network(half)
        right = odd_even_merge_sort_network(half)
        
        # Shift right comparators
        right = [(i + half, j + half) for (i, j) in right]
        
        # Merge: odd-even merge
        # First, compare (i, i + half) for i in 0..half-1
        merge_comp = [(i, i + half) for i in range(half)]
        
        # Then recursively merge odd and even subsequences
        # Odd positions: 1, 3, 5, ...
        # Even positions: 0, 2, 4, ...
        # After the first merge step, we need to sort odd and even positions
        
        # Actually, let's use a simpler approach: just use odd-even transposition
        # for now, as implementing merge sort network correctly is complex
        return left + right + merge_comp

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
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

            positions = list(range(n))
            random.shuffle(positions)

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def simulated_annealing(start_net, time_limit, seed=42):
        """
        Simulated annealing to minimize comparator count.
        Moves: remove a comparator, add a random comparator, swap two comparators.
        Energy = count + penalty * violations
        """
        random.seed(seed)
        
        current = list(start_net)
        current_valid = verify_network(current)
        current_energy = len(current) + (0 if current_valid else 100)
        
        best = list(current)
        best_energy = current_energy
        best_valid = current_valid
        
        # Temperature schedule
        t = 50.0
        alpha = 0.999
        t_min = 0.5
        
        # Generate a pool of candidate comparators for "add" moves
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        while time.time() < time_limit:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a comparator
                if len(current) == 0:
                    continue
                idx = random.randint(0, len(current) - 1)
                candidate = current[:idx] + current[idx + 1:]
            elif move_type < 0.7:
                # Add a random comparator
                pair = random.choice(all_pairs)
                # Avoid duplicates
                if pair in current:
                    continue
                candidate = current + [pair]
            else:
                # Swap two comparators (replace one with another)
                if len(current) == 0:
                    continue
                idx = random.randint(0, len(current) - 1)
                pair = random.choice(all_pairs)
                candidate = current[:idx] + [pair] + current[idx + 1:]
            
            # Evaluate candidate
            candidate_valid = verify_network(candidate)
            candidate_energy = len(candidate) + (0 if candidate_valid else 100)
            
            # Acceptance criterion
            delta = candidate_energy - current_energy
            if delta < 0:
                # Always accept improvements
                current = candidate
                current_valid = candidate_valid
                current_energy = candidate_energy
            else:
                # Accept with probability exp(-delta / t)
                if random.random() < np.exp(-delta / t):
                    current = candidate
                    current_valid = candidate_valid
                    current_energy = candidate_energy
            
            # Update best
            if current_valid and current_energy < best_energy:
                best_energy = current_energy
                best = list(current)
                best_valid = True
            
            # Cool down
            t *= alpha
            if t < t_min:
                # Restart with a higher temperature
                t = 20.0
        
        return best, best_valid

    # ---- Main search ----
    best_network = None
    best_count = float('inf')

    # Phase 1: Start from known networks and greedily prune
    candidates = [
        ("odd_even_trans", odd_even_transposition()),
        ("bubble", bubble_sort_network()),
    ]

    for name, net in candidates:
        if time.time() > deadline - 30.0:
            break
        pruned = greedy_remove(net, time_limit=start_time + 15.0)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

    # Phase 2: Simulated annealing from the best pruned network
    if best_network is not None:
        sa_net, sa_valid = simulated_annealing(
            best_network,
            time_limit=deadline - 10.0,
            seed=42
        )
        if sa_valid and len(sa_net) < best_count:
            best_count = len(sa_net)
            best_network = sa_net

    # Phase 3: Try SA from a different starting point (random dense network)
    random.seed(123)
    dense_net = []
    # Start with a moderately dense random network
    num_dense = 80
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    dense_net = all_pairs[:num_dense]
    
    if verify_network(dense_net):
        sa_net2, sa_valid2 = simulated_annealing(
            dense_net,
            time_limit=deadline - 10.0,
            seed=123
        )
        if sa_valid2 and len(sa_net2) < best_count:
            best_count = len(sa_net2)
            best_network = sa_net2

    # Phase 4: Final greedy polish
    if best_network is not None:
        final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
