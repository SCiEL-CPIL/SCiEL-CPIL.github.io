"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
TOTAL_WALL_LIMIT = 90.0
SAFETY_MARGIN = 10.0  # seconds


# EVOLVE-BLOCK-START

def _apply_network(network, inputs):
    """Apply network to a single input tuple; return output tuple."""
    vals = list(inputs)
    for i, j in network:
        a, b = vals[i], vals[j]
        if a > b:
            vals[i], vals[j] = b, a
    return tuple(vals)


def _verify_network(network):
    """Verify network sorts all 0-1 inputs using zero-one principle."""
    for mask in range(1 << N):
        # Build binary input
        inp = tuple((mask >> i) & 1 for i in range(N))
        out = _apply_network(network, inp)
        # Check sorted: non-decreasing
        for k in range(N - 1):
            if out[k] > out[k + 1]:
                return False
    return True


def _odd_even_transposition(N):
    """Classic odd-even transposition network: N(N-1)/2 comparators."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _random_network(n_comps):
    """Generate a random comparator network with n_comps comparators."""
    comps = []
    for _ in range(n_comps):
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        comps.append((i, j))
    return comps


def _greedy_prune(network):
    """Remove unnecessary comparators via greedy forward checking."""
    # Try removing each comparator and check if network still sorts
    # This is expensive, so we do a limited version
    result = list(network)
    n = len(result)
    removed = 0
    for idx in range(n - 1, -1, -1):
        if removed > 20:  # limit removals
            break
        trial = result[:idx] + result[idx + 1:]
        if _verify_network(trial):
            result = trial
            removed += 1
    return result


def _swap_mutate(network):
    """Mutate by swapping two random comparators' positions or changing a comparator."""
    net = list(network)
    if not net:
        return net
    if random.random() < 0.5 and len(net) > 1:
        # Swap two comparators
        i, j = random.sample(range(len(net)), 2)
        net[i], net[j] = net[j], net[i]
    else:
        # Change one comparator to a random new one
        idx = random.randint(0, len(net) - 1)
        a, b = random.sample(range(N), 2)
        if a > b:
            a, b = b, a
        net[idx] = (a, b)
    return net


def _insert_mutate(network):
    """Insert a random comparator at a random position."""
    net = list(network)
    a, b = random.sample(range(N), 2)
    if a > b:
        a, b = b, a
    pos = random.randint(0, len(net))
    net.insert(pos, (a, b))
    return net


def _remove_mutate(network):
    """Remove a random comparator."""
    if len(network) <= 1:
        return list(network)
    net = list(network)
    idx = random.randint(0, len(net) - 1)
    net.pop(idx)
    return net


def _crossover(net1, net2):
    """Uniform crossover of two networks."""
    length = max(len(net1), len(net2))
    child = []
    for i in range(length):
        if random.random() < 0.5:
            child.append(net1[i] if i < len(net1) else net2[i] if i < len(net2) else (0, 1))
        else:
            child.append(net2[i] if i < len(net2) else net1[i] if i < len(net1) else (0, 1))
    return child


def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_WALL_LIMIT - SAFETY_MARGIN

    # Start with the known 120-comparator odd-even transposition network
    current_best = _odd_even_transposition(N)
    best_count = len(current_best)
    assert _verify_network(current_best), "Initial network should be valid"

    random.seed(42)

    # Local search: try to remove comparators
    # Use a more efficient verification: for each candidate removal,
    # we can short-circuit. But 65536 checks is expensive in pure Python.
    # Instead, use a smarter approach: track which comparators are "redundant"
    # by checking if removing them still passes a subset of tests first.

    # Pre-compute: for the initial network, find which comparators are removable
    # Do this in batches to stay within time.

    # Strategy: iterative removal with verification
    # For speed, use a fast verification on a subset first, then full verify

    def fast_verify_subset(network, subset_mask):
        """Verify on a subset of inputs for speed."""
        for mask in subset_mask:
            inp = tuple((mask >> i) & 1 for i in range(N))
            out = _apply_network(network, inp)
            for k in range(N - 1):
                if out[k] > out[k + 1]:
                    return False
        return True

    # Create a representative subset for fast checks
    # Use all inputs with exactly k ones for k in {1, 2, 7, 8, 9, 14, 15}
    # This is much smaller than all 65536
    fast_subset = []
    for k in [1, 2, 7, 8, 9, 14, 15]:
        for comb in combinations(range(N), k):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            fast_subset.append(mask)
    # Also include some random masks
    for _ in range(500):
        fast_subset.append(random.randint(0, (1 << N) - 1))

    # Iterative removal
    network = list(current_best)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing from end to beginning (less disruptive)
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            trial = network[:idx] + network[idx + 1:]
            # Fast check first
            if fast_verify_subset(trial, fast_subset):
                # Full check
                if _verify_network(trial):
                    network = trial
                    changed = True
                    best_count = len(network)
                    break  # restart scan from end

    # Randomized search: try small mutations
    if time.time() < deadline:
        best_net = list(network)
        best_len = len(best_net)
        stall = 0
        while time.time() < deadline and stall < 50:
            # Try various mutations
            mut_net = _swap_mutate(best_net)
            if _verify_network(mut_net) and len(mut_net) < best_len:
                best_net = mut_net
                best_len = len(mut_net)
                stall = 0
            else:
                stall += 1
                # Try insertion then removal
                if len(best_net) < 200:
                    ins_net = _insert_mutate(best_net)
                    if _verify_network(ins_net):
                        # Try to prune
                        pruned = _greedy_prune(ins_net)
                        if len(pruned) < best_len:
                            best_net = pruned
                            best_len = len(pruned)
                            stall = 0
                        else:
                            stall += 1
                    else:
                        stall += 1
                else:
                    stall += 1

        if len(best_net) < len(network):
            network = best_net
            best_count = len(best_net)

    # Final verification
    assert _verify_network(network), "Final network must be valid"

    return network

# EVOLVE-BLOCK-END
