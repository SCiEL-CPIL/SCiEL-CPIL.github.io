"""Size-efficient sorting networks on 17 wires - fast local search with systematic deletion."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 82.0  # safety margin from 90s


def _build_all_inputs():
    """Build all 131072 Boolean inputs as a bit-packed uint32 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint32)


_ALL_BITS = _build_all_inputs()


def _popcount_fast(arr):
    """Fast popcount using 8-bit lookup table."""
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup[i] = bin(i).count('1')
    b0 = (arr & 0xFF).astype(np.uint8)
    b1 = ((arr >> 8) & 0xFF).astype(np.uint8)
    b2 = ((arr >> 16) & 0xFF).astype(np.uint8)
    b3 = ((arr >> 24) & 0xFF).astype(np.uint8)
    return (lookup[b0] + lookup[b1] + lookup[b2] + lookup[b3]).astype(np.uint8)


_POPCOUNTS = _popcount_fast(_ALL_BITS)

# Expected sorted output: for input with k ones, output has k ones in lowest k positions
_EXPECTED_BITS = np.zeros(NUM_INPUTS, dtype=np.uint32)
for k in range(N + 1):
    mask = (1 << k) - 1
    _EXPECTED_BITS[_POPCOUNTS == k] = mask


def _apply_comparator(bits, i, j):
    """Apply compare-exchange (i, j) to all inputs. Wire i gets min, wire j gets max."""
    bit_i = (bits >> i) & 1
    bit_j = (bits >> j) & 1
    mask = (bit_i & ~bit_j) & 1
    bits = bits & ~(mask << i)
    bits = bits & ~(mask << j)
    bits = bits | ((bit_j & mask) << i)
    bits = bits | ((bit_i & mask) << j)
    return bits


def _verify_network(comparators):
    """Verify network sorts all Boolean inputs using bit-packed representation."""
    bits = _ALL_BITS.copy()
    for (i, j) in comparators:
        bits = _apply_comparator(bits, i, j)
    return np.array_equal(bits, _EXPECTED_BITS)


def _depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    wire_depth = [0] * N
    max_d = 0
    for (i, j) in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        max_d = max(max_d, d)
        wire_depth[i] = d
        wire_depth[j] = d
    return max_d


def _odd_even_transposition():
    """Generate odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _systematic_deletion(network, deadline):
    """Systematically try to remove each comparator in order, repeating passes until no more can be removed."""
    current = list(network)
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for idx in range(len(current)):
            if time.time() >= deadline:
                return current
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                improved = True
                break  # restart from beginning after each removal
    return current


def _random_local_search(network, deadline, rng):
    """Random local search: try random removals, additions, and swaps."""
    current = list(network)
    best = list(network)
    best_count = len(best)
    
    while time.time() < deadline:
        move_type = rng.random()
        candidate = current[:]
        
        if move_type < 0.5 and len(candidate) > 20:
            # Remove random comparator
            idx = rng.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type < 0.7 and len(candidate) < 200:
            # Add random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        elif move_type < 0.85 and len(candidate) > 1:
            # Move comparator to new position
            idx = rng.randint(0, len(candidate) - 1)
            comp = candidate.pop(idx)
            new_pos = rng.randint(0, len(candidate))
            candidate.insert(new_pos, comp)
        else:
            # Swap two comparators
            if len(candidate) >= 2:
                idx1 = rng.randint(0, len(candidate) - 2)
                idx2 = rng.randint(idx1 + 1, len(candidate) - 1)
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        
        if _verify_network(candidate):
            if len(candidate) < best_count:
                best = candidate[:]
                best_count = len(best)
            current = candidate
    
    return best


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with odd-even transposition (known correct, 136 comparators)
    best = _odd_even_transposition()
    if not _verify_network(best):
        return best
    
    # Phase 1: Systematic deletion to quickly reduce size
    phase1_deadline = start_time + 15.0
    best = _systematic_deletion(best, phase1_deadline)
    
    # Phase 2: More systematic deletion passes
    phase2_deadline = start_time + 30.0
    best = _systematic_deletion(best, phase2_deadline)
    
    # Phase 3: Random local search for further improvements
    phase3_deadline = start_time + 60.0
    best = _random_local_search(best, phase3_deadline, rng)
    
    # Phase 4: Final systematic deletion
    final_deadline = start_time + DEADLINE - 2.0
    best = _systematic_deletion(best, final_deadline)
    
    # Verify final result
    if _verify_network(best):
        return best
    
    # Fallback: return odd-even transposition
    return _odd_even_transposition()
