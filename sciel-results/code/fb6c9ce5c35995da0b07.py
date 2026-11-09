"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from math import comb

N = 18
NUM_BOOLS = 1 << N  # 262144

def _build_ooe_network(n):
    """Odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _bitonic_sort(n):
    """Generate a bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort halves
    left = _bitonic_sort(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    # Merge the two sorted halves
    merge = _bitonic_merge(half)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    half = n // 2
    # Compare elements that are half apart
    comps = []
    for i in range(half):
        comps.append((i, i + half))
    # Recursively merge halves
    left = _bitonic_merge(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    return comps + left + right

def _generate_bitonic_for_n(n):
    """Generate bitonic sort network for arbitrary n by padding to next power of 2."""
    if n <= 1:
        return []
    # Find next power of 2
    p2 = 1
    while p2 < n:
        p2 *= 2
    # Generate bitonic sort for p2 wires
    full_net = _bitonic_sort(p2)
    # Remove comparators involving wires >= n
    filtered = [(i, j) for (i, j) in full_net if i < n and j < n]
    return filtered

# Precompute expected bit counts for verification
_EXPECTED_COUNTS = {}
def _get_expected_counts(n):
    if n not in _EXPECTED_COUNTS:
        counts = []
        for i in range(n):
            count = 0
            for j in range(n - i, n + 1):
                count += comb(n, j)
            counts.append(count)
        _EXPECTED_COUNTS[n] = counts
    return _EXPECTED_COUNTS[n]

def _verify_fast(comps, n=N):
    """Fast verification using Python big integers (bit-parallel)."""
    # Represent each wire's state as a Python integer where bit k indicates
    # whether wire is 1 for input k
    # Initialize: wire i has bit k set iff (k >> i) & 1
    # For wire i, the pattern repeats with period 2^(i+1), with 2^i ones then 2^i zeros
    
    # Build initial masks using pattern repetition
    # For wire i, the 64-bit word pattern depends on i
    # But using Python big ints is cleaner
    
    masks = [0] * n
    # For wire i, bit k is set iff (k >> i) & 1
    # We can build this by: for each block of 2^(i+1) bits, the upper 2^i are 1
    # Actually, let's just compute it directly for n=18 (262144 bits)
    
    # More efficient: build pattern for each wire
    # Wire i: pattern of 2^i ones followed by 2^i zeros, repeated
    # The full mask is this pattern repeated 2^n / 2^(i+1) = 2^(n-i-1) times
    
    for i in range(n):
        period = 1 << (i + 1)
        half_period = 1 << i
        # One period: lower half_period bits are 0, upper half_period bits are 1
        period_mask = ((1 << half_period) - 1) << half_period
        # Repeat this pattern to fill 2^n bits
        num_repeats = (1 << n) // period
        mask = 0
        for r in range(num_repeats):
            mask |= period_mask << (r * period)
        masks[i] = mask
    
    # Apply comparators
    for (a, b) in comps:
        # min goes to a, max goes to b
        # min = AND, max = OR
        new_a = masks[a] & masks[b]
        new_b = masks[a] | masks[b]
        masks[a] = new_a
        masks[b] = new_b
    
    # Verify: wire i should have exactly expected_counts[i] bits set
    expected = _get_expected_counts(n)
    for i in range(n):
        # Count bits in masks[i]
        # Python 3.10+ has int.bit_count()
        if hasattr(masks[i], 'bit_count'):
            bc = masks[i].bit_count()
        else:
            bc = bin(masks[i]).count('1')
        if bc != expected[i]:
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    n = N
    random.seed(42)
    
    # Generate candidate networks and pick the best starting point
    candidates = []
    
    # Odd-even transposition
    ooe = _build_ooe_network(n)
    candidates.append(ooe)
    
    # Bitonic sort (padded to 32 wires, filtered to 18)
    bitonic = _generate_bitonic_for_n(n)
    candidates.append(bitonic)
    
    # Pick the smallest valid candidate
    best_net = None
    best_count = float('inf')
    
    for cand in candidates:
        if len(cand) < best_count:
            if _verify_fast(cand, n):
                best_net = cand
                best_count = len(cand)
    
    if best_net is None:
        # Fallback: use OOE
        best_net = ooe
        best_count = len(ooe)
    
    current_net = list(best_net)
    current_count = len(current_net)
    
    # Local search: iterative removal with randomization
    # Phase 1: Greedy removal - try removing each comparator in order
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle order for diversity
        order = list(range(len(current_net)))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            if idx >= len(current_net):
                continue
            # Try removing comparator at idx
            trial = current_net[:idx] + current_net[idx+1:]
            if _verify_fast(trial, n):
                current_net = trial
                current_count -= 1
                improved = True
                break  # Restart after a removal
    
    # Phase 2: Try swapping adjacent comparators to find new redundancies
    for _round in range(200):
        if time.time() > deadline - 3.0:
            break
        found_improvement = False
        
        # Try random swaps
        num_swaps_to_try = min(50, len(current_net) - 1)
        for _ in range(num_swaps_to_try):
            if time.time() > deadline - 3.0:
                break
            if len(current_net) < 2:
                break
            
            idx = random.randint(0, len(current_net) - 2)
            # Swap
            trial = current_net[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            
            if _verify_fast(trial, n):
                # Try to remove a comparator from the swapped network
                remove_order = list(range(len(trial)))
                random.shuffle(remove_order)
                for j in remove_order[:10]:  # Try up to 10 random removals
                    if time.time() > deadline - 3.0:
                        break
                    trial2 = trial[:j] + trial[j+1:]
                    if _verify_fast(trial2, n):
                        current_net = trial2
                        current_count -= 1
                        found_improvement = True
                        break
                if found_improvement:
                    break
        
        if not found_improvement:
            # Try a few more rounds of pure removal
            removed_any = False
            order = list(range(len(current_net)))
            random.shuffle(order)
            for idx in order:
                if time.time() > deadline - 3.0:
                    break
                if idx >= len(current_net):
                    continue
                trial = current_net[:idx] + current_net[idx+1:]
                if _verify_fast(trial, n):
                    current_net = trial
                    current_count -= 1
                    removed_any = True
                    break
            if not removed_any:
                break
    
    # Phase 3: Final cleanup - one more pass of greedy removal
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        order = list(range(len(current_net)))
        random.shuffle(order)
        for idx in order:
            if time.time() > deadline - 1.0:
                break
            if idx >= len(current_net):
                continue
            trial = current_net[:idx] + current_net[idx+1:]
            if _verify_fast(trial, n):
                current_net = trial
                current_count -= 1
                improved = True
                break
    
    # Final verification
    if not _verify_fast(current_net, n):
        # Shouldn't happen, but fallback
        return best_net
    
    return current_net
# EVOLVE-BLOCK-END
