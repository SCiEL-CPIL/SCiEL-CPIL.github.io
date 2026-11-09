import time
import random

N = 17
MAX_COMPARATORS = 4096

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_fast(net):
    """Verify network sorts all 2^N Boolean inputs using big-int bit tricks."""
    total = 1 << N
    state = [0] * N
    for w in range(N):
        block = 1 << w
        val = 0
        for start in range(0, total, 2 * block):
            mask = ((1 << block) - 1) << (start + block)
            val |= mask
        state[w] = val
    
    for (i, j) in net:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
    
    for j in range(1, N):
        if state[j] & ~state[j - 1]:
            return False
    return True

def _greedy_remove(net, start_time, max_time):
    """Greedy removal: try to remove each comparator, verify, keep if valid."""
    if not _verify_fast(net):
        return net
    i = 0
    while i < len(net):
        if time.time() - start_time > max_time:
            break
        candidate = net[:i] + net[i+1:]
        if _verify_fast(candidate):
            net = candidate
        else:
            i += 1
    return net

def _randomized_greedy_remove(net, start_time, max_time, rng):
    """Randomized greedy removal: try removing in random order."""
    if not _verify_fast(net):
        return net
    n = len(net)
    indices = list(range(n))
    rng.shuffle(indices)
    removed = [False] * n
    for idx in indices:
        if time.time() - start_time > max_time:
            break
        if removed[idx]:
            continue
        candidate = [net[k] for k in range(n) if not removed[k] and k != idx]
        if _verify_fast(candidate):
            removed[idx] = True
    return [net[k] for k in range(n) if not removed[k]]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 80.0
    
    net = _seed_network()
    assert _verify_fast(net), "Seed network should be valid"
    
    best_net = list(net)
    best_len = len(net)
    
    # Phase 1: Sequential greedy removal
    net = _greedy_remove(net, start_time, max_time)
    if len(net) < best_len:
        best_net = list(net)
        best_len = len(net)
    
    # Phase 2: Multiple randomized restarts
    rng = random.Random(42)
    while time.time() - start_time < max_time - 2.0:
        current = list(best_net)
        current = _randomized_greedy_remove(current, start_time, max_time, rng)
        if len(current) < best_len:
            best_net = current
            best_len = len(current)
            current = _greedy_remove(current, start_time, max_time)
            if len(current) < best_len:
                best_net = current
                best_len = len(current)
        
        if time.time() - start_time < max_time - 5.0:
            current = _seed_network()
            current = _randomized_greedy_remove(current, start_time, max_time, rng)
            if len(current) < best_len:
                best_net = current
                best_len = len(current)
                current = _greedy_remove(current, start_time, max_time)
                if len(current) < best_len:
                    best_net = current
                    best_len = len(current)
    
    if not _verify_fast(best_net):
        return _seed_network()
    
    return best_net
# EVOLVE-BLOCK-END
