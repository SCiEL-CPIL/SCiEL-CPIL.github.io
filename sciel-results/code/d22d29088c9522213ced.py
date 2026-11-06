"""Size-efficient sorting network search on 18 wires.

Starts from a known compact network and performs greedy + simulated-annealing
removal/replacement to shrink the network while maintaining correctness over
all 2^18 Boolean inputs.
"""
import time
import random
import numpy as np

N = 18
NUM_BOOL = 1 << N  # 262144
DEADLINE = 75.0  # seconds of wall-clock budget


def _precompute_masks():
    """Precompute the bit matrix for all 2^N masks: shape (N, NUM_BOOL)."""
    masks = np.arange(NUM_BOOL, dtype=np.uint32)
    powers = (1 << np.arange(N, dtype=np.uint32))
    bits = ((masks[None, :] & powers[:, None]) != 0).astype(np.int8)
    return bits


# Precompute once
_BITS_ALL = _precompute_masks()


def _verify(net):
    """Check network sorts all 2^N boolean inputs. Returns True if valid."""
    if not net:
        return N <= 1
    bits = _BITS_ALL.copy()
    for (i, j) in net:
        bi = bits[i]
        bj = bits[j]
        bits[i] = np.minimum(bi, bj)
        bits[j] = np.maximum(bi, bj)
    # Check sorted: for each adjacent pair of wires, no inversion
    for k in range(N - 1):
        if np.any(bits[k] > bits[k + 1]):
            return False
    return True


def _verify_partial(net, sample_size=4096):
    """Fast partial verification on a random sample. For quick rejection."""
    if not net:
        return N <= 1
    # Use a random subset of masks
    indices = np.random.randint(0, NUM_BOOL, size=sample_size)
    bits = _BITS_ALL[:, indices].copy()
    for (i, j) in net:
        bi = bits[i]
        bj = bits[j]
        bits[i] = np.minimum(bi, bj)
        bits[j] = np.maximum(bi, bj)
    for k in range(N - 1):
        if np.any(bits[k] > bits[k + 1]):
            return False
    return True


def _build_seed():
    """Build the odd-even transposition sorting network (153 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _try_remove(net, idx):
    """Try removing comparator at index idx. Returns (success, new_net)."""
    new_net = net[:idx] + net[idx+1:]
    if _verify(new_net):
        return True, new_net
    return False, net


def _greedy_remove(net, max_rounds=500, time_limit=None):
    """Greedily try to remove comparators. Returns the smallest network found."""
    best = list(net)
    best_len = len(best)
    start = time.time()
    
    for round_num in range(max_rounds):
        if time_limit is not None and time.time() - start > time_limit:
            break
        progress = False
        # Try removing each comparator, starting from a random offset
        offset = random.randint(0, len(best) - 1) if best else 0
        for idx in range(len(best)):
            actual_idx = (idx + offset) % len(best)
            success, new_net = _try_remove(best, actual_idx)
            if success:
                best = new_net
                best_len -= 1
                progress = True
                break  # Restart from beginning after a removal
        if not progress:
            break
    
    return best


def _sa_optimize(net, time_budget):
    """Simulated annealing optimization. Returns the best network found."""
    best = list(net)
    best_len = len(best)
    current = list(net)
    current_len = len(current)
    
    start_time = time.time()
    T = 1.0
    T_min = 0.005
    cooling = 0.997
    
    rng = random.Random(12345)
    
    while time.time() - start_time < time_budget and T > T_min:
        if len(current) <= 68:
            break
        
        move_type = rng.random()
        
        if move_type < 0.50:
            # Removal (favored)
            idx = rng.randint(0, len(current) - 1)
            new_net = current[:idx] + current[idx+1:]
            new_len = len(current) - 1
        elif move_type < 0.85:
            # Replacement
            idx = rng.randint(0, len(current) - 1)
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            new_pair = (i, j)
            if new_pair == current[idx]:
                continue
            new_net = current[:idx] + [new_pair] + current[idx+1:]
            new_len = len(current)
        else:
            # Addition (rare)
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            new_pair = (i, j)
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [new_pair] + current[pos:]
            new_len = len(current) + 1
        
        if time.time() - start_time > time_budget:
            break
        
        # Quick check first
        if not _verify_partial(new_net, sample_size=2048):
            continue
        
        # Full check
        if _verify(new_net):
            delta = new_len - current_len
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current = new_net
                current_len = new_len
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
        T *= cooling
    
    return best


def construct():
    """Construct a sorting network for N=18 wires."""
    start_time = time.time()
    
    # Build seed
    seed = _build_seed()
    best = list(seed)
    best_len = len(best)
    
    # Phase 1: Greedy removal (most impactful)
    greedy_budget = min(15.0, (DEADLINE - (time.time() - start_time)) * 0.35)
    if greedy_budget > 2.0:
        greedy_result = _greedy_remove(best, max_rounds=1000, time_limit=greedy_budget)
        if len(greedy_result) < best_len:
            best = greedy_result
            best_len = len(best)
    
    # Phase 2: SA optimization
    elapsed = time.time() - start_time
    sa_budget = max(1.0, DEADLINE - elapsed - 5.0)
    if sa_budget > 2.0:
        sa_result = _sa_optimize(best, sa_budget)
        if len(sa_result) < best_len:
            best = sa_result
            best_len = len(best)
    
    # Phase 3: Final greedy pass
    remaining = DEADLINE - (time.time() - start_time) - 2.0
    if remaining > 1.0:
        final_result = _greedy_remove(best, max_rounds=300, time_limit=remaining)
        if len(final_result) < best_len:
            best = final_result
            best_len = len(best)
    
    # Final verification
    if not _verify(best):
        return seed
    
    return best
