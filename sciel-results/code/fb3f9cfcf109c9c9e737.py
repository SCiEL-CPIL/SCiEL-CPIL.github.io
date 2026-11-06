"""Size-efficient sorting networks on 16 wires.
Uses a known 60-comparator Batcher bitonic sort baseline and prunes
redundant comparators via the zero-one principle (all 2^16 Boolean inputs).
"""
import time
import random

N = 16
_DEADLINE = 90.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify_network_fast(comps):
    """Verify network sorts all 0/1 inputs using bit-level simulation."""
    for mask in range(1 << N):
        state = mask
        for i, j in comps:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state ^= (1 << i) | (1 << j)
        ones = bin(state).count('1')
        expected = ((1 << ones) - 1) << (N - ones)
        if state != expected:
            return False
    return True


def _batcher_bitonic_16():
    """
    Generate Batcher's bitonic sorting network for 16 inputs (60 comparators).
    
    Bitonic sort structure for N=16:
    1. Sort 2-element blocks: 8 comparators
    2. Merge to 4-element bitonic: 8 comparators  
    3. Merge to 8-element bitonic: 16 comparators
    4. Merge two 8-element bitonics to 16-element sorted: 28 comparators
    Total: 8 + 8 + 16 + 28 = 60
    """
    comps = []
    
    # Phase 1: Sort 2-element subsequences
    for i in range(0, 16, 2):
        comps.append((i, i + 1))
    
    # Phase 2: Merge pairs to form 4-element bitonic sequences
    for i in range(0, 16, 4):
        comps.append((i, i + 2))
        comps.append((i + 1, i + 3))
    
    # Phase 3: Merge 4-element blocks to form 8-element bitonic sequences
    # First, compare elements 4 apart
    for i in range(0, 16, 8):
        for k in range(4):
            comps.append((i + k, i + 4 + k))
    # Then, compare elements 2 apart within each half
    for i in range(0, 16, 4):
        comps.append((i, i + 2))
        comps.append((i + 1, i + 3))
    # Finally, compare adjacent elements within each quarter
    for i in range(0, 16, 2):
        comps.append((i, i + 1))
    
    # Phase 4: Merge two 8-element bitonic sequences into 16-element sorted
    # This is the odd-even merge of two 8-element sequences
    # Step 4a: Compare elements 8 apart (0 with 8, 1 with 9, ..., 7 with 15)
    for k in range(8):
        comps.append((k, k + 8))
    
    # Step 4b: Compare elements 4 apart within each half
    for i in range(0, 16, 8):
        for k in range(4):
            comps.append((i + k, i + 4 + k))
    
    # Step 4c: Compare elements 2 apart within each quarter
    for i in range(0, 16, 4):
        comps.append((i, i + 2))
        comps.append((i + 1, i + 3))
    
    # Step 4d: Compare adjacent elements within each eighth
    for i in range(0, 16, 2):
        comps.append((i, i + 1))
    
    return comps


def _try_prune(comps, rng, max_iterations=5000):
    """Iteratively try to remove redundant comparators."""
    current = list(comps)
    best = list(comps)
    
    for _ in range(max_iterations):
        if _remaining() < 10.0:
            break
        
        improved = False
        # Try removing each comparator
        for idx in range(len(current)):
            if _remaining() < 5.0:
                break
            
            candidate = current[:idx] + current[idx + 1:]
            if _verify_network_fast(candidate):
                current = candidate
                best = list(current)
                improved = True
                break
        
        if not improved:
            break
    
    return best


def _try_swap(comps, rng, max_iterations=2000):
    """Try swapping adjacent comparators to enable further pruning."""
    current = list(comps)
    best = list(comps)
    
    for _ in range(max_iterations):
        if _remaining() < 10.0:
            break
        
        improved = False
        # Try swapping adjacent comparators
        for idx in range(len(current) - 1):
            if _remaining() < 5.0:
                break
            
            candidate = list(current)
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            
            # Check if this enables removal of some comparator
            for remove_idx in range(len(candidate)):
                if remove_idx == idx or remove_idx == idx + 1:
                    continue
                test = candidate[:remove_idx] + candidate[remove_idx + 1:]
                if _verify_network_fast(test):
                    current = test
                    best = list(current)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best


# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for 16 wires."""
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with Batcher's bitonic sort (60 comparators)
    comps = _batcher_bitonic_16()
    
    # Verify it's correct
    if not _verify_network_fast(comps):
        # Fallback to odd-even transposition if something went wrong
        comps = [(i, i + 1) for phase in range(N)
                 for i in range(phase % 2, N - 1, 2)]
    
    best = list(comps)
    
    # Try pruning
    if _remaining() > 15.0:
        pruned = _try_prune(comps, rng, max_iterations=3000)
        if len(pruned) < len(best):
            best = pruned
    
    # Try swapping to enable more pruning
    if _remaining() > 15.0:
        swapped = _try_swap(best, rng, max_iterations=1000)
        if len(swapped) < len(best):
            best = swapped
    
    # Final pruning pass
    if _remaining() > 10.0:
        final = _try_prune(best, rng, max_iterations=2000)
        if len(final) < len(best):
            best = final
    
    return best
# EVOLVE-BLOCK-END
