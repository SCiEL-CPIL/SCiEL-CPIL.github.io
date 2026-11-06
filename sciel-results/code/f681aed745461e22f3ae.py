"""Size-efficient sorting network on 18 wires using zero-one principle + bit-parallel verification."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin (90s limit)

    # Pre-compute bit masks for verification (zero-one principle)
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

    # Pre-compute complement masks for sortedness check
    # wires[k] must be subset of wires[k+1]: check wires[k] & ~wires[k+1] == 0
    # We'll compute ~wires[k+1] on the fly

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism (zero-one principle).
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

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        Tries all positions each round until no removal works.
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
            # Try all positions in random order
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

    def odd_even_merge_network(n):
        """
        Odd-even merge network for n wires.
        Returns list of comparators.
        """
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into halves
        half = n // 2
        if n % 2 == 0:
            # Even case: split into two halves of size n/2
            left = odd_even_merge_network(half)
            right = odd_even_merge_network(half)
            # Shift right half comparators
            for (i, j) in right:
                right.append((i + half, j + half))
            # Wait, that's wrong. Let me redo.
            # Actually, the right half comparators should be on wires half..n-1
            right = [(i + half, j + half) for (i, j) in odd_even_merge_network(half)]
            
            # Merge phase: odd-even merge
            merge = _odd_even_merge(half)
            
            return left + right + merge
        else:
            # Odd case: split into sizes (n+1)//2 and (n-1)//2
            left_size = (n + 1) // 2
            right_size = (n - 1) // 2
            left = odd_even_merge_network(left_size)
            right = [(i + left_size, j + left_size) for (i, j) in odd_even_merge_network(right_size)]
            
            # Merge phase
            merge = _odd_even_merge_odd(left_size, right_size)
            
            return left + right + merge

    def _odd_even_merge(half):
        """
        Odd-even merge step for two sorted lists of size half.
        This is the merge network for two halves of size half.
        """
        comparators = []
        # The odd-even merge network for two lists of size n
        # Phase 1: compare (1,2), (3,4), ... (odd, even)
        for i in range(1, 2 * half, 2):
            if i + 1 < 2 * half:
                comparators.append((i, i + 1))
        
        # Recursively merge the even and odd subsequences
        # Even subsequence: wires 0, 2, 4, ..., 2*half-2
        # Odd subsequence: wires 1, 3, 5, ..., 2*half-1
        # Each subsequence has size half
        even_merge = _odd_even_merge(half // 2) if half >= 2 else []
        # Map even_merge indices: wire 2*i in the merged space
        for (i, j) in even_merge:
            comparators.append((2 * i, 2 * j))
        
        odd_merge = _odd_even_merge(half // 2) if half >= 2 else []
        # Map odd_merge indices: wire 2*i + 1
        for (i, j) in odd_merge:
            comparators.append((2 * i + 1, 2 * j + 1))
        
        # Phase 2: compare (0,1), (2,3), ... (even, odd)
        for i in range(0, 2 * half - 1, 2):
            comparators.append((i, i + 1))
        
        return comparators

    def _odd_even_merge_odd(left_size, right_size):
        """
        Odd-even merge for two lists of different sizes.
        Simplified: just use a standard merge network approximation.
        """
        # For simplicity, use a basic merge structure
        comparators = []
        total = left_size + right_size
        # Just do a simple odd-even transposition on the combined
        for phase in range(total):
            for i in range(phase % 2, total - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def shell_sort_network():
        """
        Generate a shell sort inspired network.
        """
        comparators = []
        # Use various gaps
        gaps = [9, 5, 3, 1]
        for gap in gaps:
            for i in range(gap, N):
                comparators.append((i - gap, i))
        return comparators

    def simulated_annealing(initial_net, t_max=500, t_min=0.1, alpha=0.999, max_iter=50000, deadline=None):
        """
        Simulated Annealing to minimize comparator count.
        Aggressive removal bias.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            
            if r < 0.5:
                # Remove (aggressive)
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif r < 0.65:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
            elif r < 0.80:
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
                continue  # Invalid, reject

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                # Always accept improvement
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                # Accept with probability exp(-delta / t)
                delta = new_energy - current_energy
                if delta > 0:
                    prob = np.exp(-delta / t)
                    if random.random() < prob:
                        current = candidate
                        current_energy = new_energy
            
            t *= alpha
            if t < t_min:
                break
        
        return best

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    candidates.append(("shell", shell_sort_network()))
    
    # Add odd-even merge if it works
    try:
        oem = odd_even_merge_network(N)
        if verify_network(oem):
            candidates.append(("oem", oem))
    except:
        pass
    
    # Add some random dense starts
    for i in range(3):
        random.seed(200 + i)
        comparators = []
        for _ in range(120):
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                comparators.append((u, v))
        random.shuffle(comparators)
        candidates.append((f"random_{i}", comparators))

    best_network = None
    best_count = float('inf')

    # Run SA on each candidate
    for name, net in candidates:
        if time.time() > deadline - 10.0:
            break
        
        # First, try to prune the initial network to get a better starting point for SA
        prune_limit = start_time + 8.0
        pruned = greedy_remove(net, time_limit=prune_limit)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        sa_deadline = deadline - 8.0
        sa_result = simulated_annealing(
            pruned, 
            t_max=300, 
            t_min=0.05, 
            alpha=0.9995, 
            max_iter=30000, 
            deadline=sa_deadline
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
