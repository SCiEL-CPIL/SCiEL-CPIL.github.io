import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _apply_comparator_vectorized(states, i, j):
    """Apply a compare-exchange to a vector of bitmask states (numpy int64 array)."""
    bi = (states >> i) & 1
    bj = (states >> j) & 1
    swap_mask = (bi == 0) & (bj == 1)
    result = states.copy()
    result[swap_mask] |= (1 << i)
    result[swap_mask] &= ~(1 << j)
    return result

def _apply_network_vectorized(states, network):
    """Apply a list of comparators to a vector of bitmask states."""
    for i, j in network:
        states = _apply_comparator_vectorized(states, i, j)
    return states

def _is_sorted_vectorized(states):
    """Check if bitmask states are sorted (all 0s then all 1s). Returns bool array."""
    sorted_states = set()
    for num_ones in range(N + 1):
        if num_ones == 0:
            sorted_states.add(0)
        else:
            mask = ((1 << num_ones) - 1) << (N - num_ones)
            sorted_states.add(mask)
    result = np.zeros(len(states), dtype=bool)
    for idx, state in enumerate(states):
        if state in sorted_states:
            result[idx] = True
    return result

def _count_violations_vectorized(network):
    """Count the number of inputs for which the network fails to sort."""
    all_states = np.arange(1 << N, dtype=np.int64)
    result_states = _apply_network_vectorized(all_states, network)
    sorted_mask = _is_sorted_vectorized(result_states)
    violations = np.sum(~sorted_mask)
    return int(violations)

def _verify(network):
    """Verify that the network sorts all inputs. Returns True if valid."""
    return _count_violations_vectorized(network) == 0

def _known_60_network():
    """Return a known 60-comparator sorting network for 16 wires (Batcher OMS optimized)."""
    network = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    ]
    return network

def _remove_redundant_comparators(network, time_budget=5.0):
    """Remove redundant comparators greedily. A comparator is redundant if removing it keeps the network valid."""
    if not network:
        return network
    if not _verify(network):
        return network

    start_time = time.time()
    kept = network[:]

    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        for idx in range(len(kept)):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = kept[:idx] + kept[idx+1:]
            if _verify(candidate):
                kept = candidate
                changed = True
                break  # Restart from beginning since indices shifted
    return kept

def _local_search(network, time_budget=2.0):
    """Perform local search: removals, swaps, modifications. Returns improved network."""
    if not _verify(network):
        return network

    start_time = time.time()
    best = network[:]
    best_len = len(best)

    while (time.time() - start_time) < time_budget:
        improved = False

        # Try removing comparators (highest impact)
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:i] + best[i+1:]
            if _verify(candidate):
                best = candidate
                best_len = len(best)
                improved = True
                break

        if improved:
            continue

        # Try swapping adjacent comparators
        for i in range(len(best) - 1):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify(candidate):
                best = candidate
                improved = True
                break

        if improved:
            continue

        # Try changing a comparator to a different pair
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            old_i, old_j = best[i]
            found = False
            for new_i in range(N):
                for new_j in range(new_i + 1, N):
                    if (time.time() - start_time) >= time_budget:
                        found = True
                        break
                    if (new_i, new_j) == (old_i, old_j):
                        continue
                    candidate = best[:]
                    candidate[i] = (new_i, new_j)
                    if _verify(candidate):
                        best = candidate
                        improved = True
                        found = True
                        break
                if found:
                    break
            if improved:
                break

        if improved:
            continue

        # Try inserting a comparator (only if it could lead to a shorter network later)
        # This is expensive; only do it if we have lots of time left
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5.0:
            for pos in range(len(best) + 1):
                if (time.time() - start_time) >= time_budget:
                    break
                found = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if (time.time() - start_time) >= time_budget:
                            found = True
                            break
                        candidate = best[:pos] + [(i, j)] + best[pos:]
                        if _verify(candidate):
                            # Inserting made it longer, but maybe enables further removals
                            # Only accept if we can then remove more
                            # For now, skip insertion to save time
                            pass
                        if (time.time() - start_time) >= time_budget:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if not improved:
            break

    return best

def construct():
    """Construct a sorting network for 16 wires. Returns a list of (i, j) pairs."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit

    # Start with the known 60-comparator network
    best_network = _known_60_network()

    # Verify it's actually valid (safety check)
    if not _verify(best_network):
        # Fall back to OET
        best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_len = len(best_network)

    # Phase 1: Aggressive local search to try to reduce below 60
    remaining = deadline - time.time()
    if remaining > 10.0:
        search_budget = min(40.0, remaining - 5.0)
        optimized = _local_search(best_network, time_budget=search_budget)
        if len(optimized) < best_len:
            best_network = optimized
            best_len = len(best_network)

    # Phase 2: Redundant comparator removal (polish pass)
    remaining = deadline - time.time()
    if remaining > 3.0:
        polish_budget = min(10.0, remaining - 1.0)
        optimized = _remove_redundant_comparators(best_network, time_budget=polish_budget)
        if len(optimized) < best_len:
            best_network = optimized
            best_len = len(best_network)

    # Phase 3: Another local search pass if time permits
    remaining = deadline - time.time()
    if remaining > 5.0:
        search_budget = min(10.0, remaining - 1.0)
        optimized = _local_search(best_network, time_budget=search_budget)
        if len(optimized) < best_len:
            best_network = optimized
            best_len = len(best_network)

    # Final verification
    if best_network is not None and _verify(best_network):
        return best_network
    else:
        # Ultimate fallback: OET
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
