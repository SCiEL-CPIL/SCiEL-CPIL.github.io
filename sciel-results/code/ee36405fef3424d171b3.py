"""Size-efficient sorting networks on 17 wires."""
import numpy as np
import time

N = 17
DEADLINE = 85.0  # safety margin before 90s kill
_start_time = None

# EVOLVE-BLOCK-START

def _time_left():
    if _start_time is None:
        return DEADLINE
    return DEADLINE - (time.time() - _start_time)

def _odd_even_merge(a, b, offset):
    """Merge two sorted halves of length a and b (a>=b) starting at offset."""
    comps = []
    if b == 0:
        return comps
    if a == b:
        # Interleave
        for i in range(b):
            comps.append((offset + i, offset + b + i))
        # Then recursively merge the two interleaved halves
        # Actually for equal sizes, use the standard OEM
        if b == 1:
            return comps
        # Split into pairs
        comps += _odd_even_merge(1, 1, offset)
        return comps
    # Standard odd-even merge
    # Split b into even and odd positions
    b_even = b // 2
    b_odd = b - b_even
    # Merge a's even positions with b's even positions
    a_even = (a + 1) // 2
    a_odd = a - a_even
    comps += _odd_even_merge(a_even, b_even, offset)
    # Merge a's odd positions with b's odd positions
    comps += _odd_even_merge(a_odd, b_odd, offset + 1)
    # Interleave
    for i in range(min(a_even, b_even)):
        comps.append((offset + 2*i, offset + 2*i + 1))
    for i in range(min(a_odd, b_odd)):
        comps.append((offset + 2*i + 1, offset + 2*i + 2))
    return comps

def _make_odd_even_merge_network(n):
    """Build odd-even merge sort network for n wires."""
    comps = []
    def rec(size, offset):
        if size == 1:
            return
        half = size // 2
        rec(half, offset)
        rec(size - half, offset + half)
        comps.extend(_odd_even_merge(half, size - half, offset))
    rec(n, 0)
    return comps

def _make_transposition_network(n):
    """Odd-even transposition network (seed)."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify(comps):
    """Verify sorting network using zero-one principle with numpy."""
    n = N
    # Generate all 2^n boolean inputs
    # For n=17, that's 131072 inputs - manageable
    all_inputs = np.arange(2**n, dtype=np.uint32)
    # Create matrix of bits: shape (2^n, n)
    # bit i of input goes to wire i
    bits = np.zeros((2**n, n), dtype=np.uint8)
    for i in range(n):
        bits[:, i] = (all_inputs >> i) & 1
    
    # Apply comparators
    for (i, j) in comps:
        a = bits[:, i]
        b = bits[:, j]
        # min goes to i, max goes to j
        bits[:, i] = np.minimum(a, b)
        bits[:, j] = np.maximum(a, b)
    
    # Check: sorted means non-decreasing: bits[:, k] <= bits[:, k+1] for all k
    # For boolean, this means no pattern 1,0
    diffs = bits[:, 1:].astype(np.int8) - bits[:, :-1].astype(np.int8)
    return np.all(diffs >= 0)

def _verify_sample(comps, num_samples=10000):
    """Quick verification with random samples for speed during search."""
    n = N
    rng = np.random.default_rng(42)
    # Test all-ones, all-zeros, and random
    for _ in range(num_samples):
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        for (i, j) in comps:
            a = bits[i]
            b = bits[j]
            bits[i] = min(a, b)
            bits[j] = max(a, b)
        # Check sorted
        for k in range(n - 1):
            if bits[k] > bits[k+1]:
                return False
    return True

def _delete_pass(comps, num_passes=30, verify_fn=None):
    """Try to delete each comparator if the network still sorts."""
    if verify_fn is None:
        verify_fn = _verify
    comps = list(comps)
    for _ in range(num_passes):
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            trial = comps[:idx] + comps[idx+1:]
            if verify_fn(trial):
                comps = trial
                changed = True
    return comps

def _local_search(comps, max_iters=5000, verify_fn=None):
    """Local search: try moving comparators to different positions, swapping, etc."""
    if verify_fn is None:
        verify_fn = _verify
    comps = list(comps)
    best_len = len(comps)
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        if len(comps) >= best_len:
            break
        
        # Try random modifications
        move_type = rng.integers(0, 3)
        trial = list(comps)
        
        if move_type == 0:
            # Move a comparator to a random position
            if len(trial) > 1:
                src = rng.integers(0, len(trial))
                dst = rng.integers(0, len(trial))
                if src != dst:
                    elem = trial.pop(src)
                    trial.insert(dst, elem)
        elif move_type == 1:
            # Swap two comparators
            if len(trial) > 1:
                i = rng.integers(0, len(trial))
                j = rng.integers(0, len(trial))
                if i != j:
                    trial[i], trial[j] = trial[j], trial[i]
        else:
            # Delete a random comparator
            if len(trial) > 1:
                idx = rng.integers(0, len(trial))
                trial.pop(idx)
        
        if verify_fn(trial) and len(trial) < best_len:
            comps = trial
            best_len = len(comps)
    
    return comps

def _greedy_add_remove(comps, verify_fn=None):
    """Try to find a better network by local operations."""
    if verify_fn is None:
        verify_fn = _verify
    comps = list(comps)
    
    # Multiple deletion passes
    for _ in range(30):
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            trial = comps[:idx] + comps[idx+1:]
            if verify_fn(trial):
                comps = trial
                changed = True
        if not changed:
            break
    
    return comps

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with odd-even merge sort network
    comps = _make_odd_even_merge_network(N)
    print(f"OEM network: {len(comps)} comparators")
    
    # Verify it's correct
    assert _verify(comps), "OEM network failed verification!"
    
    best_comps = list(comps)
    best_len = len(comps)
    
    # Aggressive deletion passes
    for i in range(30):
        if _time_left() < 15.0:
            break
        changed = False
        for idx in range(len(best_comps) - 1, -1, -1):
            trial = best_comps[:idx] + best_comps[idx+1:]
            if _verify(trial):
                best_comps = trial
                changed = True
        if not changed:
            break
        best_len = len(best_comps)
        print(f"Deletion pass {i}: {best_len} comparators")
    
    # Local search
    if _time_left() > 20.0:
        best_comps = _local_search(best_comps, max_iters=3000, verify_fn=_verify)
        best_len = len(best_comps)
        print(f"After local search: {best_len} comparators")
    
    # More deletion after local search
    if _time_left() > 10.0:
        for i in range(20):
            if _time_left() < 5.0:
                break
            changed = False
            for idx in range(len(best_comps) - 1, -1, -1):
                trial = best_comps[:idx] + best_comps[idx+1:]
                if _verify(trial):
                    best_comps = trial
                    changed = True
            if not changed:
                break
            best_len = len(best_comps)
            print(f"Post-LB deletion pass {i}: {best_len} comparators")
    
    print(f"Final: {len(best_comps)} comparators")
    return best_comps

# EVOLVE-BLOCK-END
