import time
import numpy as np

N = 18
TIME_LIMIT = 55  # Safety margin

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized state fill using bit patterns
    # For each block of 64 inputs, generate the bit pattern for each wire
    for block in range(num_blocks):
        base = block * 64
        for w in range(N):
            # For inputs base..base+63, wire w is set if bit w of input is 1
            # Input k has bit w set if (k >> w) & 1 == 1
            # For consecutive inputs base..base+63, the pattern of bit w is:
            # period = 2^(w+1), half_period = 2^w
            # Within each period, first half_period have bit 0, next half_period have bit 1
            period = 1 << (w + 1)
            half = 1 << w
            # Determine which of the 64 positions in this block have bit w set
            # Position p in block corresponds to input base + p
            # Bit w of (base + p) is set if ((base + p) >> w) & 1 == 1
            # We can compute this for all 64 positions at once
            mask = np.zeros(64, dtype=np.uint8)
            for p in range(64):
                if ((base + p) >> w) & 1:
                    mask[p] = 1
            # Convert mask to uint64
            val = 0
            for p in range(64):
                if mask[p]:
                    val |= (1 << p)
            states[w][block] = val

    def _fast_verify(comparators):
        # Use a working copy to avoid modifying original states
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted: for all w, if wire w has a 1, wire w+1 must have a 1
        # i.e., current[w] & ~current[w+1] must be all zeros
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net, time_budget=None):
        # Try to remove comparators; use a time budget if provided
        if time_budget is None:
            time_budget = deadline - 5
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            # Try removals from the end first, then from the beginning
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > time_budget:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    break  # Restart from end after a removal
            if not changed:
                # Also try from the beginning
                for idx in range(len(net)):
                    if time.time() > time_budget:
                        return net
                    trial = net[:idx] + net[idx+1:]
                    if _fast_verify(trial):
                        net = trial
                        changed = True
                        break
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search with multiple move types and simulated annealing
    temperature = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline - 10:
        # Move 1: Swap adjacent
        if len(current_network) > 1:
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    temperature = 0.1  # Reset temperature on improvement

        # Move 2: Mutation (change wire pair) - bias toward nearby wires
        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            # Pick new pair - bias toward nearby wires for better locality
            if np.random.rand() < 0.7:
                # Nearby wire
                a = max(0, min(N - 1, i + np.random.randint(-3, 4)))
                b = max(0, min(N - 1, j + np.random.randint(-3, 4)))
            else:
                a = np.random.randint(N)
                b = np.random.randint(N)
            if a == b:
                b = (a + 1) % N
            if a > b: a, b = b, a
            if a == i and b == j: continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    temperature = 0.1

        # Move 3: Insert-Delete (2-opt style)
        if len(current_network) < 200 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network) + 1)
            a = np.random.randint(N)
            b = np.random.randint(N)
            if a == b:
                b = (a + 1) % N
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a,b)] + current_network[idx:]
            reduced = greedy_reduce(trial)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    temperature = 0.1

        # Move 4: Swap two non-adjacent comparators (position swap)
        if len(current_network) > 2 and time.time() < deadline - 10:
            idx1 = np.random.randint(len(current_network))
            idx2 = np.random.randint(len(current_network))
            if idx1 != idx2:
                trial = current_network.copy()
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                if _fast_verify(trial):
                    current_network = trial
                    reduced = greedy_reduce(current_network)
                    if len(reduced) < len(current_network):
                        current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
                        temperature = 0.1

        # Move 5: Block swap - swap two blocks of comparators
        if len(current_network) > 4 and time.time() < deadline - 10:
            block_size = np.random.randint(1, max(2, len(current_network) // 4))
            idx1 = np.random.randint(0, len(current_network) - block_size)
            idx2 = np.random.randint(0, len(current_network) - block_size)
            if abs(idx1 - idx2) > block_size:  # Non-overlapping
                trial = current_network.copy()
                block1 = trial[idx1:idx1+block_size]
                block2 = trial[idx2:idx2+block_size]
                trial[idx1:idx1+block_size] = block2
                trial[idx2:idx2+block_size] = block1
                if _fast_verify(trial):
                    current_network = trial
                    reduced = greedy_reduce(current_network)
                    if len(reduced) < len(current_network):
                        current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
                        temperature = 0.1

        # Cool down
        temperature *= cooling_rate

    # Final polish pass: try to remove more comparators with more aggressive search
    if time.time() < deadline - 3:
        final_net = list(best_network)
        # Multiple passes of greedy reduction
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for idx in range(len(final_net) - 1, -1, -1):
                    if time.time() > deadline - 3:
                        break
                    trial = final_net[:idx] + final_net[idx+1:]
                    if _fast_verify(trial):
                        final_net = trial
                        changed = True
                        break
                if not changed:
                    for idx in range(len(final_net)):
                        if time.time() > deadline - 3:
                            break
                        trial = final_net[:idx] + final_net[idx+1:]
                        if _fast_verify(trial):
                            final_net = trial
                            changed = True
                            break
        if len(final_net) < best_len:
            best_network = final_net
            best_len = len(final_net)

    return best_network
