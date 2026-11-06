import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute all 2^N boolean inputs as int32
all_masks = np.arange(1 << N, dtype=np.int32)

# Precompute bit masks for each wire
BIT_MASKS = np.array([1 << i for i in range(N)], dtype=np.int32)

# Precompute popcounts
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
W0_bytes = all_masks[:, None] // (256 ** np.arange(4)) % 256
popcounts = popcount_table[W0_bytes].sum(axis=1)

# Precompute target sorted values
targets = np.zeros(N + 1, dtype=np.int32)
for k in range(N + 1):
    if k == 0:
        targets[k] = 0
    else:
        targets[k] = (1 << N) - (1 << (N - k))

expected_sorted = targets[popcounts]


def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0


def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses fast vectorized NumPy operations.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return False

    W = all_masks.copy()

    for i, j in net:
        bi = BIT_MASKS[i]
        bj = BIT_MASKS[j]

        # Extract bits
        ai = (W & bi) != 0
        aj = (W & bj) != 0

        # min goes to wire i, max goes to wire j
        mn = ai & aj
        mx = ai | aj

        # Clear bits i and j, then set new values
        clear_mask = ~(bi | bj)
        W = (W & clear_mask) | (mn.astype(np.int32) * bi) | (mx.astype(np.int32) * bj)

    return np.array_equal(W, expected_sorted)


def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Start with odd-even transposition network
    net = _odd_even_transposition()

    best_net = net[:]
    best_count = len(net)

    # Phase 1: Greedy removal - try removing each comparator
    changed = True
    while changed:
        if time.time() > deadline - 20:
            break
        changed = False

        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 20:
                return best_net

            new_net = net[:idx] + net[idx + 1:]

            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break

        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 20:
                    return best_net

                new_net = net[:idx] + net[idx + 1:]

                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break

    # Phase 2: Local search with simulated annealing
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.05
        cooling = 0.9995

        current_count = len(net)

        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0

            # Generate move
            r = random.random()
            if r < 0.35:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx + 1:]
            elif r < 0.55:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            elif r < 0.75:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            elif r < 0.88:
                # Swap
                if len(net) < 2:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            else:
                # Move
                if len(net) == 0:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                comp = new_net.pop(idx1)
                new_net.insert(idx2, comp)

            if not _is_sorted_network(new_net):
                continue

            new_count = len(new_net)
            delta = new_count - current_count

            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count

            T *= cooling

    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()

    return best_net
