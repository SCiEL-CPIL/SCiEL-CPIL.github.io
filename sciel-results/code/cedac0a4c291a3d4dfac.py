import time
import random
import numpy as np

N = 17
TOTAL_INPUTS = 1 << N
DEADLINE = time.time() + 85  # 5s safety margin before 90s kill

def _simulate_network(network, inputs):
    """Simulate network on batch of inputs. inputs: (num_inputs, N) array of 0/1.
    Returns (num_inputs, N) sorted outputs."""
    x = inputs.copy()
    for i, j in network:
        xi = x[:, i]
        xj = x[:, j]
        x[:, i] = np.minimum(xi, xj)
        x[:, j] = np.maximum(xi, xj)
    return x

def _is_valid(network):
    """Check if network sorts all 2^N Boolean inputs (zero-one principle)."""
    # Generate all 2^N inputs
    if N <= 20:
        # Use bit manipulation to generate all inputs
        all_inputs = np.arange(TOTAL_INPUTS, dtype=np.uint32)
        # Expand to binary
        bits = ((all_inputs[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
        # Simulate
        result = _simulate_network(network, bits)
        # Check each row is sorted: non-decreasing
        # sorted means result[:, k] <= result[:, k+1] for all k
        diffs = np.diff(result, axis=1)
        if np.any(diffs < 0):
            return False
        return True
    return False

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    # Assign each comparator to a layer based on wire availability
    wire_last_layer = [-1] * N
    depth = 0
    for i, j in network:
        d = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = d
        wire_last_layer[j] = d
        depth = max(depth, d)
    return depth

def _make_seed_network():
    """Create initial network - odd-even transposition for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _try_remove(network, idx):
    """Try removing comparator at index idx, check validity."""
    new_network = network[:idx] + network[idx+1:]
    if _is_valid(new_network):
        return new_network
    return None

def _try_insert(network, i, j):
    """Try inserting comparator (i,j) at best position, check validity."""
    if i >= j or i < 0 or j >= N:
        return None
    # Try inserting at various positions
    best_pos = None
    for pos in range(len(network) + 1):
        new_network = network[:pos] + [(i, j)] + network[pos:]
        if _is_valid(new_network):
            return new_network
    return None

def construct():
    """Construct a sorting network for N=17 wires, minimizing comparator count."""
    start_time = time.time()
    
    # Initial network
    network = _make_seed_network()
    best_network = list(network)
    best_count = len(best_network)
    
    # Verify seed
    if not _is_valid(best_network):
        # Fallback: should not happen
        return best_network
    
    # Phase 1: Greedy removal - repeatedly try to remove comparators
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 10:
        improved = False
        # Shuffle order to escape local minima
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if (time.time() - start_time) > DEADLINE - 10:
                break
            new_net = _try_remove(network, idx)
            if new_net is not None:
                network = new_net
                improved = True
                if len(network) < best_count:
                    best_count = len(network)
                    best_network = list(network)
    
    # Phase 2: Local search - try swapping adjacent comparators
    if (time.time() - start_time) < DEADLINE - 15:
        for _ in range(50):
            if (time.time() - start_time) > DEADLINE - 15:
                break
            if len(network) < 2:
                break
            # Try swapping adjacent comparators
            idx = random.randint(0, len(network) - 2)
            new_net = list(network)
            new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            if _is_valid(new_net):
                network = new_net
    
    # Phase 3: Try inserting useful comparators to enable further removals
    if (time.time() - start_time) < DEADLINE - 20:
        for _ in range(20):
            if (time.time() - start_time) > DEADLINE - 20:
                break
            # Try inserting a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            if (i, j) in network:
                continue
            new_net = _try_insert(network, i, j)
            if new_net is not None:
                # Now try to remove some comparators
                temp_net = new_net
                temp_improved = True
                while temp_improved and (time.time() - start_time) < DEADLINE - 10:
                    temp_improved = False
                    indices = list(range(len(temp_net)))
                    random.shuffle(indices)
                    for idx in indices:
                        if (time.time() - start_time) > DEADLINE - 10:
                            break
                        rem = _try_remove(temp_net, idx)
                        if rem is not None:
                            temp_net = rem
                            temp_improved = True
                if len(temp_net) < len(network):
                    network = temp_net
                    if len(network) < best_count:
                        best_count = len(network)
                        best_network = list(network)
    
    # Phase 4: Final greedy removal pass
    if (time.time() - start_time) < DEADLINE - 5:
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 5:
            improved = False
            indices = list(range(len(network)))
            random.shuffle(indices)
            for idx in indices:
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                new_net = _try_remove(network, idx)
                if new_net is not None:
                    network = new_net
                    improved = True
                    if len(network) < best_count:
                        best_count = len(network)
                        best_network = list(network)
    
    # Return the best network found
    if len(best_network) <= len(network):
        return best_network
    else:
        return network
