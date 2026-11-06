import time
import numpy as np

N = 18
TIME_LIMIT = 55  # Safety margin for 90s limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Pre-compute bit-parallel states for fast verification
    # Number of 64-bit blocks needed for 2^18 inputs
    num_blocks = 2**N // 64

    # states[w] is an array of num_blocks uint64s
    # Bit b of states[w][k] is the value of wire w for input (k*64 + b)
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]

    # Fill states efficiently using vectorized bit extraction
    # For each wire w, bit w of input index k determines the value
    # Process in chunks to avoid memory issues
    chunk_size = 4096
    for start in range(0, 2**N, chunk_size):
        end = min(start + chunk_size, 2**N)
        indices = np.arange(start, end, dtype=np.uint64)
        for w in range(N):
            bits = (indices >> w) & 1
            # Reshape to (num_blocks_in_chunk, 64) and pack into uint64
            n = len(bits)
            full_blocks = n // 64
            rem = n % 64
            if full_blocks > 0:
                packed = bits[:full_blocks * 64].reshape(full_blocks, 64)
                # Pack 64 bits into a single uint64: bit 0 of block = LSB
                # np.packbits packs MSB first by default, so we reverse
                packed = np.packbits(packed[:, ::-1], axis=1).astype(np.uint64)
                states[w][start // 64 : start // 64 + full_blocks] = packed
            if rem > 0:
                rem_bits = bits[full_blocks * 64:]
                # Pad to 64
                padded = np.zeros(64, dtype=np.uint8)
                padded[:rem] = rem_bits
                packed = np.packbits(padded[::-1], axis=0).astype(np.uint64)
                states[w][start // 64 + full_blocks] = packed

    def _fast_verify(comparators):
        """
        Verify network using bit-parallel simulation.
        Returns True if network sorts all inputs.
        """
        # Copy states
        current = [s.copy() for s in states]

        for i, j in comparators:
            a = current[i]
            b = current[j]
            # min goes to i, max goes to j
            current[i] = a & b
            current[j] = a | b

        # Check if sorted
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)

    # Verify seed
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)

    def greedy_reduce(net):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 5:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    # Do not increment idx, check the new element at idx
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)

    # Phase 2: Simulated Annealing + Local Search
    np.random.seed(42)

    # Temperature schedule for simulated annealing
    T = 5.0  # Initial temperature (in comparators)
    T_min = 0.5
    cooling_rate = 0.97

    while time.time() < deadline - 8:
        if len(current_network) > 1:
            # Choose a random move type
            move_type = np.random.random()

            if move_type < 0.3 and len(current_network) > 1:
                # Random Swap: swap two adjacent comparators
                idx = np.random.randint(len(current_network) - 1)
                trial = current_network.copy()
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                if _fast_verify(trial):
                    current_network = trial
                    current_network = greedy_reduce(current_network)
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)

            elif move_type < 0.5:
                # Wire Mutation: change the wires of one comparator
                idx = np.random.randint(len(current_network))
                i, j = current_network[idx]
                a, b = np.random.choice(N, 2, replace=False)
                if a > b:
                    a, b = b, a
                if a == i and b == j:
                    continue
                trial = current_network.copy()
                trial[idx] = (a, b)
                if _fast_verify(trial):
                    current_network = trial
                    current_network = greedy_reduce(current_network)
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)

            elif move_type < 0.7 and len(current_network) > 0:
                # Deletion: try removing a random comparator
                idx = np.random.randint(len(current_network))
                trial = current_network[:idx] + current_network[idx+1:]
                if _fast_verify(trial):
                    current_network = trial
                    current_network = greedy_reduce(current_network)
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
                else:
                    # Simulated annealing: accept with probability exp(-delta/T)
                    delta = 1.0  # Size increases by 1 when we don't remove
                    # Actually, deletion failed, so we keep current_network
                    # But we could try accepting a worse state temporarily
                    # For now, just skip
                    pass

            elif move_type < 0.85:
                # Insertion: add a random comparator at a random position
                if len(current_network) < 200:  # Cap to avoid explosion
                    pos = np.random.randint(len(current_network) + 1)
                    a, b = np.random.choice(N, 2, replace=False)
                    if a > b:
                        a, b = b, a
                    trial = current_network[:pos] + [(a, b)] + current_network[pos:]
                    if _fast_verify(trial):
                        # Accept even though it's larger (annealing)
                        current_network = trial
                        current_network = greedy_reduce(current_network)
                        if len(current_network) < best_len:
                            best_network = list(current_network)
                            best_len = len(current_network)

            else:
                # 2-opt: remove two comparators and add one new one
                if len(current_network) > 2:
                    idx1 = np.random.randint(len(current_network))
                    idx2 = np.random.randint(len(current_network))
                    if idx1 == idx2:
                        continue
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    trial = current_network[:idx1] + current_network[idx1+1:idx2] + current_network[idx2+1:]
                    # Add a new random comparator
                    pos = np.random.randint(len(trial) + 1)
                    a, b = np.random.choice(N, 2, replace=False)
                    if a > b:
                        a, b = b, a
                    trial = trial[:pos] + [(a, b)] + trial[pos:]
                    if _fast_verify(trial):
                        current_network = trial
                        current_network = greedy_reduce(current_network)
                        if len(current_network) < best_len:
                            best_network = list(current_network)
                            best_len = len(current_network)

        # Cool down
        T = max(T_min, T * cooling_rate)

        # Periodically restart from best if stuck
        if len(current_network) == best_len and T < 2.0:
            # Try a fresh greedy reduction from best
            current_network = list(best_network)
            current_network = greedy_reduce(current_network)
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
            T = 3.0  # Reheat

    # Final greedy reduction
    current_network = greedy_reduce(best_network)
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)

    return best_network
# EVOLVE-BLOCK-END
