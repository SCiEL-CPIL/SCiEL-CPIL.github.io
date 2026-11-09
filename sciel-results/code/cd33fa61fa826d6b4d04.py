"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 17
_DEADLINE = 80  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def _apply_compare(mask, i, j):
    """Apply compare-exchange on wires i,j: min to i, max to j.
    mask is an integer where bit k represents wire k (1 = high/1)."""
    vi = (mask >> i) & 1
    vj = (mask >> j) & 1
    if vi > vj:
        # swap: set i to 1, j to 0
        mask |= (1 << i)
        mask &= ~(1 << j)
    elif vi < vj:
        # swap: set i to 0, j to 1
        mask &= ~(1 << i)
        mask |= (1 << j)
    return mask

def _verify(network):
    """Verify sorting network using zero-one principle.
    Returns True if network sorts all 2^N binary inputs."""
    for mask in range(1 << N):
        for i, j in network:
            vi = (mask >> i) & 1
            vj = (mask >> j) & 1
            if vi > vj:
                mask |= (1 << i)
                mask &= ~(1 << j)
            elif vi < vj:
                mask &= ~(1 << i)
                mask |= (1 << j)
        # After sorting, all 1s should be on the right (higher indices)
        # So the mask should be of form: 0...0 1...1
        # Check: if mask != 0, the lowest set bit should be at position (N - popcount)
        if mask == 0:
            continue
        # Count trailing zeros
        tz = 0
        m = mask
        while (m & 1) == 0:
            tz += 1
            m >>= 1
        # The sorted form should have exactly (N - tz) ones starting from position tz
        expected = ((1 << (N - tz)) - 1) << tz
        if mask != expected:
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed
    if not _verify(network):
        return best
    
    # Greedy removal: try removing each comparator and check if still valid
    # Process in reverse order to preserve structure where possible
    for attempt in range(len(network) - 1, -1, -1):
        if time.time() - start_time > _DEADLINE:
            break
        # Try removing comparator at position 'attempt'
        candidate = list(network)
        removed = candidate.pop(attempt)
        if _verify(candidate):
            network = candidate
            best = list(candidate)
            best_len = len(best)
    
    # Second pass: try again with different removal order
    if time.time() - start_time < _DEADLINE - 2:
        for attempt in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            removed = candidate.pop(attempt)
            if _verify(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
    
    # Third pass: try forward removal
    if time.time() - start_time < _DEADLINE - 2:
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            removed = candidate.pop(i)
            if _verify(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
    
    # Fourth pass: another reverse pass
    if time.time() - start_time < _DEADLINE - 2:
        for attempt in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            removed = candidate.pop(attempt)
            if _verify(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
    
    # Fifth pass: forward again
    if time.time() - start_time < _DEADLINE - 2:
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            removed = candidate.pop(i)
            if _verify(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
    
    return best
# EVOLVE-BLOCK-END
