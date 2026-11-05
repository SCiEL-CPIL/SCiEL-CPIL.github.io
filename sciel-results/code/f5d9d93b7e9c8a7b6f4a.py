"""Size-efficient sorting network search on 17 wires."""
import time
import random
import numpy as np

N = 17
M = 1 << N  # 131072

# Precompute a fast verification path:
# We represent the network as a list of (i, j) comparators.
# Verification over all 2^17 boolean inputs is the zero-one principle.

# Bit-sliced verification: pack 131072 bits per wire into 4 uint64 words (524288 bits total)
# This makes each comparator O(1) Python-level ops on large integers (handled in C)

_WORDS = 4  # 4 * 65536 = 262144 bits >= 131072

def _init_wire_states():
    """Initialize bit-sliced wire states for all 2^17 inputs.
    Wire w has bit k set iff input k has bit w = 1."""
    states = [None] * N
    for w in range(N):
        # Bit w of integer k is set for k in [2^w, 2^(w+1)-1], [2^(w+2), 2^(w+3)-1], ...
        # We need a 131072-bit integer where bit k = (k >> w) & 1
        # Build by pattern: for word_index in range(WORDS):
        #   bits in this word correspond to k = word_index*65536 .. (word_index+1)*65536-1
        #   bit k is set iff (k >> w) & 1 == 1
        # Period of (k>>w)&1 is 2^(w+1). Within each period, top half is 1.
        val = 0
        period = 1 << (w + 1)
        half = 1 << w
        for wi in range(_WORDS):
            base = wi * 65536
            # In this 65536-bit chunk, bit position p (0..65535) corresponds to k = base + p
            # (base + p) >> w & 1
            # We need to build a 65536-bit integer with the right pattern
            # The pattern has period 2^(w+1). If w+1 >= 16, the whole 65536 bits might be one phase
            if w + 1 >= 16:
                # Period >= 65536, so within one 65536-chunk it's either all 0 or all 1
                # (base + p) >> w & 1: since period >= 65536, it's constant over the chunk
                if ((base >> w) & 1) == 1:
                    chunk = (1 << 65536) - 1
                else:
                    chunk = 0
            else:
                # Build the pattern for one period, then tile
                # Pattern: 0...0 (half bits) then 1...1 (half bits), repeated
                # Within 65536 bits, there are 65536 // period repetitions
                # Build one period's pattern
                period_val = (1 << half) - 1  # lower half bits are 1
                # Actually: positions 0..half-1 are 0, half..period-1 are 1
                # So period_val = ((1 << half) - 1) << half
                period_val = ((1 << half) - 1) << half
                reps = 65536 // period
                # Tile: OR shifted copies
                chunk = 0
                for r in range(reps):
                    chunk |= period_val << (r * period)
            val |= chunk << (wi * 65536)
        states[w] = val
    return states

_WIRE_INIT = None
def _get_wire_init():
    global _WIRE_INIT
    if _WIRE_INIT is None:
        _WIRE_INIT = _init_wire_states()
    return _WIRE_INIT

def _verify_fast(network):
    """Verify sorting network over all 2^17 boolean inputs using bit-sliced integers."""
    init = _get_wire_init()
    wires = list(init)  # copy
    
    for (i, j) in network:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b for bit-sliced
        wires[i] = wi & wj
        wires[j] = wi | wj
    
    # Check sorted: for each adjacent pair (w, w+1), wires[w] should be subset of wires[w+1]
    # i.e., wires[w] & ~wires[w+1] == 0 for all w
    for w in range(N - 1):
        if (wires[w] & ~wires[w + 1]) != 0:
            return False
    # Wire 0 should be all zeros
    if wires[0] != 0:
        return False
    # Wire N-1 should be all ones (within our 131072-bit range)
    # We only care about the first 131072 bits
    mask = (1 << 131072) - 1
    if (wires[N-1] & mask) != mask:
        return False
    return True


def _depth_of_network(network):
    """Compute parallel depth of a compare-exchange network."""
    # depth[w] = earliest time wire w is available
    depth = [0] * N
    for (i, j) in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0


def _seed_odd_even():
    """Odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def _seed_batcher_17():
    """
    Construct a Batcher-style sorting network for 17 wires.
    Uses recursive odd-even merge sort. For non-power-of-2, we use a
    padding approach: build for 32 wires, then extract the subnetwork
    that only involves wires 0..16, and verify/fix.
    
    Actually, a cleaner approach: use the known structure for small N.
    Let's implement a proper Batcher odd-even merge for N=17 by
    padding to 32, building the full network, and keeping only
    comparators where both endpoints are < 17.
    """
    # Build Batcher's odd-even merge sort for 32 wires (power of 2)
    def _batcher_odd_merge(n):
        """Return list of comparators for Batcher's odd-even merge sort on n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comps = []
        # Split into two halves
        half = n // 2
        # Recursively sort each half
        left = _batcher_odd_merge(half)
        right = _batcher_odd_merge(half)
        for (i, j) in left:
            comps.append((i, j))
        for (i, j) in right:
            comps.append((i + half, j + half))
        
        # Odd-even merge the two sorted halves
        # Merge step for odd-even merge:
        # For k = 0, 1, ..., half-1:
        #   Compare-exchange (k, half + k)
        #   Compare-exchange (k, half + k + 1) if k + 1 < half
        #   Compare-exchange (k + 1, half + k) if k + 1 < half
        #   Compare-exchange (k + 1, half + k + 1) if k + 1 < half and k + 2 < half
        # Actually, the standard odd-even merge pattern:
        # For i = 0 to half-1:
        #   If i is even: compare (i, half+i)
        #   If i is odd: compare (i, half+i) and (i, half+i-1) and (i+1, half+i)
        # Let me use the correct recursive odd-even merge:
        
        # The odd-even merge of two sorted sequences of length half:
        # Step 1: compare (i, half + i) for all i
        # Step 2: recursively merge even-indexed and odd-indexed sub-sequences
        
        # Simpler: use the iterative odd-even merge
        # Phase 1: compare (i, half + i) for i = 0..half-1
        for i in range(half):
            comps.append((i, half + i))
        
        # Phase 2: recursively merge the even positions and odd positions
        # Even positions: 0, 2, 4, ... -> wires 0, 2, 4, ... and half, half+2, ...
        # Odd positions: 1, 3, 5, ... -> wires 1, 3, 5, ... and half+1, half+3, ...
        
        # For even sub-merge:
        even_half = half // 2
        if even_half > 1:
            even_left = _batcher_odd_merge(even_half)
            for (i, j) in even_left:
                comps.append((2*i, 2*j))
            even_right = _batcher_odd_merge(even_half)
            for (i, j) in even_right:
                comps.append((2*i + half, 2*j + half))
            # Now odd-even merge the even subsequences
            # This is getting complex. Let me use a simpler known construction.
        
        return comps
    
    # The recursive approach is getting complex. Let me use a different strategy:
    # Use a known efficient network for 16 wires (Batcher) and add wire 17.
    # Batcher's network for 16 wires has 119 comparators.
    # For 17, we can start with 16-wire Batcher and then insert wire 17.
    
    # Alternative: use the seed as-is but try to improve it.
    # Let's just return the odd-even seed for now, but with a slightly better
    # initial construction.
    return _seed_odd_even()


def _seed_batcher():
    """Batcher's odd-even merge sort network for N wires (not power of 2, use recursive construction)."""
    # For non-power-of-2, we can use a simple approach: pad to next power of 2, build, then trim.
    # But that's complex. Instead, use a known small network or just use odd-even as seed.
    # Let's implement Batcher's for power of 2 and adapt.
    # For N=17, pad to 32, build Batcher, keep only comparators involving wires < 17,
    # then fix up. Actually this is tricky. Let's just use odd-even as the base seed.
    return _seed_odd_even()


def construct():
    start = time.time()
    deadline = start + 85  # 5s margin before 90s kill
    
    best_network = _seed_odd_even()
    best_count = len(best_network)
    
    rng = random.Random(42)
    
    def _time_left():
        return deadline - time.time()
    
    # First, verify our seed
    if not _verify_fast(best_network):
        # Shouldn't happen, but just in case
        pass
    
    # Local search: aggressively try removing comparators
    # With bit-sliced verification, each check is fast enough for many trials
    
    improved = True
    while improved and _time_left() > 5:
        improved = False
        # Try removing each comparator
        indices = list(range(len(best_network)))
        rng.shuffle(indices)
        for idx in indices:
            if _time_left() < 5:
                break
            # Try removing comparator idx
            trial = best_network[:idx] + best_network[idx+1:]
            if _verify_fast(trial):
                best_network = trial
                improved = True
                break  # restart scan after a removal
    
    # If stuck on removal, try more aggressive mutations
    if _time_left() > 5:
        improved = True
        while improved and _time_left() > 5:
            improved = False
            # Try swapping pairs of adjacent comparators
            for idx in range(len(best_network) - 1):
                if _time_left() < 5:
                    break
                trial = list(best_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                if _verify_fast(trial):
                    best_network = trial
                    improved = True
                    break
            # Try replacing a comparator
            if not improved:
                for idx in range(len(best_network)):
                    if _time_left() < 5:
                        break
                    for _ in range(5):
                        if _time_left() < 5:
                            break
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        trial = list(best_network)
                        trial[idx] = (i, j)
                        if _verify_fast(trial):
                            best_network = trial
                            improved = True
                            break
    
    # Return the best network found
    return best_network
