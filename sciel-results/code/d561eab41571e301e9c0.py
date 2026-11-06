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
    
    num_blocks = 2**N // 64  # 4096
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized state fill: for each wire w, bit b of states[w][block] is set
    # if input (block*64 + b) has bit w set, i.e., (block*64 + b) >> w & 1
    # For w < 6: pattern repeats every 2^w blocks within the 64-bit word
    # For w >= 6: the bit is constant across all 64 inputs in a block, 
    #   determined by (block >> (w-6)) & 1
    for w in range(N):
        if w < 6:
            # Period is 2^(w+1) in terms of input index, so within 64 consecutive inputs
            # the pattern has period 2^(w+1)
            period = 1 << (w + 1)
            half = 1 << w
            # Create one period's worth of bits: first half are 0, second half are 1
            # But we need bit b set if (b >> w) & 1, i.e., b in [half, period)
            # Build a 64-bit mask where bits b with (b>>w)&1 are set
            mask = 0
            for b in range(64):
                if (b >> w) & 1:
                    mask |= (1 << b)
            # For each block, the pattern is the same (since period divides 64 for w<6)
            # Wait, period = 2^(w+1). For w=0, period=2. For w=5, period=64.
            # The pattern within each 64-input block is identical.
            states[w][:] = mask
        else:
            # w >= 6: bit w of input (block*64 + b) = (block*64 + b) >> w & 1
            # = ((block << 6) + b) >> w & 1
            # Since b < 64 = 2^6, and w >= 6:
            # (block << 6) >> w = block >> (w-6)
            # b >> w = 0 (since b < 64 <= 2^w for w>=6)
            # So bit = (block >> (w-6)) & 1
            # For each block, all 64 bits are the same
            shift = w - 6
            for block in range(num_blocks):
                if (block >> shift) & 1:
                    states[w][block] = 0xFFFFFFFFFFFFFFFF
                else:
                    states[w][block] = 0

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net, time_budget=0.5):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net) and time.time() < deadline - 5:
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    rng = np.random.RandomState(42)
    
    # Local Search with multiple move types
    while time.time() < deadline - 8:
        # Move 1: Swap adjacent comparators
        if len(current_network) > 1 and time.time() < deadline - 8:
            idx = rng.randint(len(current_network) - 1)
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

        # Move 2: Mutation (change wire pair at random position)
        if len(current_network) > 0 and time.time() < deadline - 8:
            idx = rng.randint(len(current_network))
            i, j = current_network[idx]
            a, b = rng.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            if a == i and b == j:
                continue
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
                    
        # Move 3: Insert-Delete (add a comparator, then try to reduce)
        if len(current_network) < 200 and time.time() < deadline - 8:
            idx = rng.randint(len(current_network) + 1)
            a, b = rng.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
            reduced = greedy_reduce(trial)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

        # Move 4: Double swap (swap two non-adjacent comparators)
        if len(current_network) > 3 and time.time() < deadline - 8:
            idx1, idx2 = rng.choice(len(current_network), 2, replace=False)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
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

    return best_network
