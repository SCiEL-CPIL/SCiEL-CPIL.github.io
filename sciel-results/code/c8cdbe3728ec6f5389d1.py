"""
Size-efficient sorting network for 18 wires.

Strategy:
  1. Build a dense Batcher odd-even mergesort network (next power of 2 = 32,
     then restrict to 18 wires by filtering comparators that touch wires >= 18).
     This gives a much larger pool of redundant comparators than the lean
     odd-even transposition seed, so greedy removal can push the count lower.
  2. Fast bit-parallel verifier over all 2^18 Boolean inputs.
  3. Greedy deletion (multiple passes, randomised order).
  4. Simulated annealing with slide / swap / delete / insert moves to escape
     local minima, always keeping the best verified network.
  5. Time-budgeted with an explicit deadline guard.
"""

import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# Precompute all Boolean inputs as a bit matrix: shape (NUM_INPUTS, N)
_INPUTS = np.arange(NUM_INPUTS, dtype=np.uint32)
_INPUTS_BITS = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUTS_BITS[:, _i] = (_INPUTS >> _i) & 1

# Precompute a mask for sorted check: for each adjacent pair, the "bad" pattern
# is x[k]==0 and x[k+1]==1. We'll check on the fly.


def _apply_network(net, inputs):
    """
    Apply the compare-exchange network to all inputs simultaneously.
    net: list of (i, j)
    inputs: (NUM_INPUTS, N) uint8 array
    Returns: sorted (NUM_INPUTS, N) uint8 array
    """
    x = inputs.copy()
    for (i, j) in net:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = a & b
        x[:, j] = a | b
    return x


def _is_sorted(x):
    """Check if each row is sorted (non-decreasing)."""
    for k in range(N - 1):
        bad = (x[:, k] == 0) & (x[:, k + 1] == 1)
        if np.any(bad):
            return False
    return True


def _verify(net):
    """Verify network sorts all 2^18 Boolean inputs."""
    x = _apply_network(net, _INPUTS_BITS)
    return _is_sorted(x)


def _batcher_odd_even_mergesort(n):
    """
    Batcher odd-even mergesort for n wires (n should be a power of 2).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    # Recursively sort first half and second half
    net = []
    net.extend(_batcher_odd_even_mergesort(n // 2))
    # Shift second half
    net2 = []
    for (i, j) in _batcher_odd_even_mergesort(n // 2):
        net2.append((i + n // 2, j + n // 2))
    net.extend(net2)
    # Odd-even merge the two halves
    net.extend(_batcher_odd_even_merge(n))
    return net


def _batcher_odd_even_merge(n):
    """
    Odd-even merge for n wires (n power of 2), assuming two sorted halves of n/2.
    """
    if n == 2:
        return [(0, 1)]
    half = n // 2
    net = []
    # Merge odd positions and even positions recursively
    # Odd: positions 1, 3, 5, ..., n-1
    # Even: positions 0, 2, 4, ..., n-2
    for (i, j) in _batcher_odd_even_merge(half):
        net.append((2 * i + 1, 2 * j + 1))
    for (i, j) in _batcher_odd_even_merge(half):
        net.append((2 * i, 2 * j))
    # Cross comparators
    for i in range(half):
        if 2 * i + 1 < n:
            net.append((2 * i, 2 * i + 1))
    return net


def _batcher_18():
    """
    Build Batcher odd-even mergesort for 32 wires, then restrict to first 18.
    Only keep comparators where both endpoints are < 18.
    """
    full_net = _batcher_odd_even_mergesort(32)
    net = [(i, j) for (i, j) in full_net if i < N and j < N]
    return net


def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def _greedy_reduce(net, deadline):
    """
    Repeatedly try to delete each comparator (in random order) and keep if still valid.
    """
    net = list(net)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return net
            candidate = net[:idx] + net[idx + 1:]
            if _verify(candidate):
                net = candidate
                improved = True
                break  # restart since indices shifted
    return net


def _simulated_annealing(net, deadline, max_iters=200000):
    """
    Simulated annealing with moves: delete, insert, swap, slide.
    Always keeps best verified network.
    """
    best_net = list(net)
    best_len = len(best_net)
    current_net = list(net)
    current_len = len(current_net)
    
    T = 50.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random()
    
    iters = 0
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = rng.random()
        candidate = list(current_net)
        
        if move_type < 0.30 and len(candidate) > 1:
            # Delete a random comparator
            idx = rng.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type < 0.55:
            # Insert a random comparator at a random position
            i, j = sorted(rng.sample(range(N), 2))
            pos = rng.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        elif move_type < 0.80:
            # Swap two comparators
            if len(candidate) >= 2:
                i, j = sorted(rng.sample(range(len(candidate)), 2))
                candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            # Slide: move a comparator to a random position
            if len(candidate) >= 2:
                idx = rng.randint(0, len(candidate) - 1)
                comp = candidate.pop(idx)
                new_pos = rng.randint(0, len(candidate))
                candidate.insert(new_pos, comp)
        
        # Verify candidate
        if _verify(candidate):
            new_len = len(candidate)
            # Accept if better or with probability based on temperature
            if new_len <= current_len:
                current_net = candidate
                current_len = new_len
                if new_len < best_len:
                    best_net = list(candidate)
                    best_len = new_len
            else:
                # Worse, accept with probability exp(-delta/T)
                delta = new_len - current_len
                if rng.random() < np.exp(-delta / max(T, 1e-6)):
                    current_net = candidate
                    current_len = new_len
        else:
            # Invalid: reject, but with small probability accept to explore
            if rng.random() < 0.05:
                current_net = candidate
                current_len = len(candidate)
        
        T = max(T_min, T * cooling)
    
    return best_net


def construct():
    """
    Construct a sorting network for 18 wires with as few comparators as possible.
    Returns: ordered list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Start with Batcher mergesort restricted to 18 wires (dense, many redundant comps)
    best_net = _batcher_18()
    
    # Verify initial network
    if not _verify(best_net):
        # Fallback to odd-even transposition if Batcher restriction failed
        best_net = _odd_even_transposition()
    
    best_len = len(best_net)
    
    # Phase 1: Greedy deletion (multiple passes)
    best_net = _greedy_reduce(best_net, deadline)
    best_len = len(best_net)
    
    if time.time() > deadline:
        return best_net
    
    # Phase 2: Simulated annealing to escape local minima
    # Run SA for a while, then greedy reduce again, repeat
    sa_iters = 0
    while time.time() < deadline - 5.0:
        # Run SA
        sa_net = _simulated_annealing(best_net, deadline - 5.0, max_iters=50000)
        if len(sa_net) < best_len:
            best_net = sa_net
            best_len = len(best_net)
        
        # Greedy reduce after SA
        best_net = _greedy_reduce(best_net, deadline - 5.0)
        best_len = len(best_net)
        
        sa_iters += 1
        if sa_iters >= 20:
            break
    
    # Final greedy pass with remaining time
    best_net = _greedy_reduce(best_net, deadline)
    
    return best_net
