import numpy as np
import time
import random

N = 17
MAX_COMPARATORS = 4096

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # Pre-compute all 2^N boolean inputs
    # inputs shape: (2^N, N), dtype uint8
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1

    def simulate(network):
        """Simulate the network on all inputs, return final state."""
        state = inputs.copy()
        for i, j in network:
            si = state[:, i]
            sj = state[:, j]
            min_val = np.minimum(si, sj)
            max_val = np.maximum(si, sj)
            state[:, i] = min_val
            state[:, j] = max_val
        return state

    def is_sorted(state):
        """Check if all rows are sorted (non-decreasing)."""
        # For 0/1 values, sorted means no 1,0 adjacent pair
        # i.e., state[:, k+1] >= state[:, k] for all k
        return np.all(state[:, 1:] >= state[:, :-1])

    def is_valid(network):
        """Check if the network sorts all 2^N inputs."""
        state = simulate(network)
        return is_sorted(state)

    # Generate initial network: Odd-Even Transposition Network
    def generate_oetn():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

    # Better initial network: AKS-style or known small networks
    # Let's start with OETN and aggressively optimize
    network = generate_oetn()

    # Verify initial network
    if not is_valid(network):
        # Fallback: identity network (shouldn't happen)
        return network

    best_network = network[:]
    best_len = len(network)

    # Strategy 1: Greedy removal - try to remove each comparator
    def try_removals(net, max_time):
        """Try to remove redundant comparators. Returns (new_net, removed_count)."""
        improved = True
        total_removed = 0
        while improved and time.time() < max_time:
            improved = False
            for idx in range(len(net)):
                if time.time() >= max_time:
                    break
                candidate = net[:idx] + net[idx+1:]
                if is_valid(candidate):
                    net = candidate
                    total_removed += 1
                    improved = True
                    break
        return net, total_removed

    # Strategy 2: Swap adjacent comparators to find better ordering
    def try_swaps(net, max_time, max_swaps=100):
        """Try swapping adjacent comparators. Returns (new_net, swaps_done)."""
        swaps_done = 0
        for _ in range(max_swaps):
            if time.time() >= max_time:
                break
            if len(net) < 2:
                break
            idx = random.randint(0, len(net) - 2)
            candidate = net[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if is_valid(candidate):
                # Check if this swap is beneficial (e.g., reduces some measure)
                # For now, just accept any valid swap
                net = candidate
                swaps_done += 1
        return net, swaps_done

    # Strategy 3: Remove pairs of comparators
    def try_pair_removals(net, max_time):
        """Try removing pairs of comparators. Returns (new_net, pairs_removed)."""
        pairs_removed = 0
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(net)):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, len(net)):
                    if time.time() >= max_time:
                        break
                    # Try removing both net[i] and net[j]
                    candidate = [net[k] for k in range(len(net)) if k != i and k != j]
                    if is_valid(candidate):
                        net = candidate
                        pairs_removed += 1
                        improved = True
                        break
                if improved:
                    break
        return net, pairs_removed

    # Strategy 4: Local search with random restarts on removal
    def local_search(net, max_time, strategy='remove'):
        """Perform local search. Returns best network found."""
        best = net[:]
        best_len = len(net)
        
        current = net[:]
        
        while time.time() < max_time:
            if strategy == 'remove':
                # Try random removal
                if len(current) <= 1:
                    break
                idx = random.randint(0, len(current) - 1)
                candidate = current[:idx] + current[idx+1:]
                if is_valid(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
            elif strategy == 'swap':
                # Try random swap
                if len(current) < 2:
                    break
                i = random.randint(0, len(current) - 2)
                j = random.randint(i + 1, len(current) - 1)
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if is_valid(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
        
        return best

    # Phase 1: Greedy removal (most impactful)
    network, removed = try_removals(network, deadline - 10.0)
    if len(network) < best_len:
        best_network = network[:]
        best_len = len(network)

    # Phase 2: More aggressive removal with pair removals
    if time.time() < deadline - 5.0:
        network2, pairs = try_pair_removals(network, deadline - 5.0)
        if len(network2) < best_len:
            best_network = network2[:]
            best_len = len(network2)

    # Phase 3: Local search with random restarts
    if time.time() < deadline - 1.0:
        # Try multiple local searches from different starting points
        for attempt in range(5):
            if time.time() >= deadline - 1.0:
                break
            
            # Start from current best or a perturbed version
            if attempt == 0:
                start_net = best_network[:]
            else:
                # Perturb: add a random comparator and try to remove
                start_net = best_network[:]
                if len(start_net) < MAX_COMPARATORS:
                    # Add random comparator
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    start_net.append((i, j))
                    # Now try to remove back to valid
                    start_net, _ = try_removals(start_net, deadline - 1.0)
            
            # Local search
            result = local_search(start_net, deadline - 1.0, strategy='remove')
            if len(result) < best_len:
                best_network = result[:]
                best_len = len(result)
        
        # Also try swap-based local search
        if time.time() < deadline - 1.0:
            result = local_search(best_network[:], deadline - 1.0, strategy='swap')
            if len(result) < best_len:
                best_network = result[:]
                best_len = len(result)

    # Final verification
    if not is_valid(best_network):
        # Fallback to original OETN
        return generate_oetn()

    return best_network
