import time
import numpy as np

N = 18
NUM_MASKS = 1 << N  # 262144

def _build_mask_matrix():
    """Precompute the (NUM_MASKS, N) boolean matrix of all 0-1 inputs."""
    masks = np.arange(NUM_MASKS, dtype=np.uint64)
    # x[m, i] = bit i of mask m
    bits = np.arange(N, dtype=np.uint64)
    mat = ((masks[:, None] >> bits[None, :]) & 1).astype(np.uint8)
    return mat

def _apply_network_vec(network, X):
    """Apply comparison network to (num_masks, N) uint8 matrix X (values 0/1).
    For 0/1: min = AND, max = OR. min to i, max to j."""
    X = X.copy()
    for i, j in network:
        a = X[:, i]
        b = X[:, j]
        X[:, i] = a & b   # min
        X[:, j] = a | b   # max
    return X

def _is_sorted_network_vec(network, MASKS):
    """Check if network sorts all 0-1 inputs. Returns (bool, num_violations)."""
    X = _apply_network_vec(network, MASKS)
    # sorted means each row is non-decreasing: X[:,k] <= X[:,k+1]
    # violation if X[:,k] == 1 and X[:,k+1] == 0
    viol = 0
    for k in range(N - 1):
        viol += int(np.sum(X[:, k] & (1 - X[:, k + 1])))
    return viol == 0, viol

def _apply_network_list(network, x):
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _depth(network):
    # longest path depth via topological layers
    # simple: compute depth of each wire
    depth = [0] * N
    for i, j in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = max(depth[i], d)
        depth[j] = max(depth[j], d)
    return max(depth) if depth else 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s

    MASKS = _build_mask_matrix()

    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]

    # Verify initial
    ok, _ = _is_sorted_network_vec(network, MASKS)
    assert ok, "Initial network invalid"

    # Phase 1: Greedy removal to get a good starting point (fast with vectorized check)
    improved = True
    while improved and (time.time() - start_time) < deadline - 25:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 25:
                break
            test = network[:idx] + network[idx + 1:]
            ok, _ = _is_sorted_network_vec(test, MASKS)
            if ok:
                network = test
                improved = True

    best_network = list(network)
    best_len = len(network)
    current_network = list(network)
    current_len = len(network)

    # Phase 2: Simulated Annealing with rich move set
    # Moves: remove, swap two comparators, insert new, relocate
    import random
    rng = random.Random(42)

    T = 1.0
    T_min = 0.01
    cooling = 0.999

    # Precompute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_index = {p: k for k, p in enumerate(all_pairs)}
    num_pairs = len(all_pairs)

    def make_move(net):
        """Generate a random move on the network. Returns new network + move type."""
        op = rng.random()
        n = len(net)
        if n == 0:
            # must insert
            p = rng.choice(all_pairs)
            return net + [p], 'insert'
        if op < 0.35 and n > 1:
            # remove
            idx = rng.randrange(n)
            return net[:idx] + net[idx + 1:], 'remove'
        elif op < 0.60:
            # swap two comparators
            if n < 2:
                p = rng.choice(all_pairs)
                return net + [p], 'insert'
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 == i2:
                i2 = (i2 + 1) % n
            new = net[:]
            new[i1], new[i2] = net[i2], net[i1]
            return new, 'swap'
        elif op < 0.80:
            # insert a new comparator at random position
            p = rng.choice(all_pairs)
            pos = rng.randrange(n + 1)
            return net[:pos] + [p] + net[pos:], 'insert'
        else:
            # relocate: remove one, insert at new position (possibly different pair)
            idx = rng.randrange(n)
            p = net[idx]
            base = net[:idx] + net[idx + 1:]
            # maybe change the pair
            if rng.random() < 0.5:
                p = rng.choice(all_pairs)
            pos = rng.randrange(len(base) + 1)
            return base[:pos] + [p] + base[pos:], 'relocate'

    # SA loop
    best_score = current_len  # minimize length
    sa_start = time.time()
    iters = 0
    no_improve = 0
    improve_threshold = 2000  # restart cooling if stuck

    while (time.time() - start_time) < deadline - 8:
        new_net, mtype = make_move(current_network)
        if len(new_net) > 4096:
            continue
        ok, viol = _is_sorted_network_vec(new_net, MASKS)
        if ok:
            new_len = len(new_net)
            new_score = new_len
            if new_score < best_score:
                best_score = new_score
                best_network = list(new_net)
                no_improve = 0
                # accept
                current_network = new_net
                current_len = new_len
                iters += 1
                continue
            # accept with probability based on delta
            delta = new_score - current_len
            if delta <= 0:
                current_network = new_net
                current_len = new_len
            else:
                if T > T_min:
                    prob = np.exp(-delta / max(T, 1e-6))
                    if rng.random() < prob:
                        current_network = new_net
                        current_len = new_len
            no_improve += 1
        else:
            # invalid: penalize, may accept if very close
            # use violation count as soft penalty
            delta = current_len + viol * 0.5 - current_len  # not useful; just reject
            # occasionally accept near-valid to escape
            if viol <= 2 and T > 0.2 and rng.random() < 0.05:
                current_network = new_net
                current_len = len(new_net)
            no_improve += 1

        iters += 1
        if no_improve > improve_threshold:
            # restart: revert to best, reheat
            current_network = list(best_network)
            current_len = best_score
            T = max(T * 1.5, 0.5)
            no_improve = 0

        T = max(T * cooling, T_min)

    # Final: greedy removal on best to shave more
    final_net = list(best_network)
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        for idx in range(len(final_net) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 2:
                break
            test = final_net[:idx] + final_net[idx + 1:]
            ok, _ = _is_sorted_network_vec(test, MASKS)
            if ok:
                final_net = test
                improved = True

    # Final verification
    ok, _ = _is_sorted_network_vec(final_net, MASKS)
    if not ok:
        # fall back to best_network
        final_net = list(best_network)
        ok, _ = _is_sorted_network_vec(final_net, MASKS)
        if not ok:
            # ultimate fallback: original odd-even
            final_net = [(i, i + 1) for phase in range(N)
                         for i in range(phase % 2, N - 1, 2)]

    return final_net
