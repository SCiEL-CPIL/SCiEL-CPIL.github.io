"""Rewritten program: size-efficient sorting network for N=18 wires.

Strategy:
  1. Start from a known-good construction (odd-even transposition, 153 comps).
  2. Use bitwise-vectorized verification over all 2^18 Boolean inputs.
  3. Apply a two-phase local search:
       Phase A: Greedy deletion – remove any comparator whose removal keeps the network sorting.
       Phase B: Simulated annealing – accept worse moves with probability exp(-Δ/T) to escape
                local minima of the deletion landscape.
  4. Time-budgeted: hard deadline at 80 s (10 s safety margin under the 90 s kill).
  5. Return the best (fewest-comparator) valid network found.

The key insight from prior runs: bitwise verification is 5-10× faster than
per-input simulation, enabling many more search steps within the time budget.
"""

import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144
DEADLINE = time.time() + 80.0  # 80 s hard stop (10 s margin)


def _apply_comparator(state, i, j):
    """Apply one compare-exchange on wires i,j to a (NUM_INPUTS, N) uint8 array.
    state[k, w] = 0/1 for input k, wire w.
    """
    a = state[:, i]
    b = state[:, j]
    # min goes to i, max goes to j
    state[:, i] = np.minimum(a, b)
    state[:, j] = np.maximum(a, b)


def _verify_network(comparators):
    """Return True if the network sorts all 2^N Boolean inputs.
    Uses bitwise (vectorized) simulation.
    """
    # Build all 2^N inputs as a (2^N, N) uint8 array
    # Wire w of input k = bit w of k
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    state = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        state[:, w] = ((indices >> w) & 1).astype(np.uint8)

    for (i, j) in comparators:
        _apply_comparator(state, i, j)

    # Check: for each input, the output must be sorted (non-decreasing across wires)
    # i.e., state[k, 0] <= state[k, 1] <= ... <= state[k, N-1]
    # Equivalent: no adjacent pair is inverted
    for w in range(N - 1):
        if np.any(state[:, w] > state[:, w + 1]):
            return False
    return True


def _verify_network_fast(comparators):
    """Faster verification using precomputed bit-sliced representation.
    Instead of (NUM_INPUTS, N), we store N bit-vectors of length NUM_INPUTS.
    Each comparator operates on two bit-vectors with bitwise AND/OR.
    """
    # bit[w] is a uint64 array of shape (NUM_INPUTS // 64,) representing
    # the values on wire w across all inputs.
    # Actually simpler: use numpy uint8 arrays of length NUM_INPUTS.
    # Precompute initial state: wire w = bit w of index
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Store as list of N arrays, each of shape (NUM_INPUTS,) uint8
    wires = [((indices >> w) & 1).astype(np.uint8) for w in range(N)]

    for (i, j) in comparators:
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)

    # Check sortedness: for each adjacent pair, no inversion
    for w in range(N - 1):
        if np.any(wires[w] > wires[w + 1]):
            return False
    return True


def _verify_bitwise(comparators):
    """Bitwise verification using 64-bit packed representation for speed.
    Represent each wire as an array of uint64 chunks.
    NUM_INPUTS = 262144 = 4096 chunks of 64 bits.
    """
    NUM_CHUNKS = NUM_INPUTS // 64  # 4096
    # Precompute initial wire values as uint64 chunks
    # Wire w: bit w of each index 0..NUM_INPUTS-1
    # Index k is in chunk k//64, bit position k%64
    # For wire w, chunk c: bit b = 1 iff (c*64 + b) has bit w set
    wires = [np.zeros(NUM_CHUNKS, dtype=np.uint64) for _ in range(N)]
    for w in range(N):
        for c in range(NUM_CHUNKS):
            # For chunk c, bits 0..63 correspond to indices c*64 .. c*64+63
            # Bit w of index = 1 when (index >> w) & 1 == 1
            # In a 64-bit word, bit w is periodic with period 2^(w+1)
            # Pattern: 0...0 1...1 repeating, block size 2^w
            # For 64-bit: if w >= 6, all bits are 1 (since 2^w >= 64)
            if w >= 6:
                wires[w][c] = 0xFFFFFFFFFFFFFFFF
            else:
                # Build the 64-bit pattern for bit w
                # Bit w is 1 for positions where (pos & (1<<w)) != 0
                # In a 64-bit word, this is a repeating pattern of 2^w zeros, 2^w ones
                block = 1 << w
                # Create pattern: for each bit position p in 0..63, bit p = (p >> w) & 1
                # We can build this:
                pattern = 0
                for p in range(64):
                    if (p >> w) & 1:
                        pattern |= (1 << p)
                wires[w][c] = pattern

    for (i, j) in comparators:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b  # min of two 0/1 values = AND
        wires[j] = a | b  # max of two 0/1 values = OR

    # Check sortedness: for each adjacent wire pair, wire[w] should be <= wire[w+1]
    # i.e., no position where wire[w]=1 and wire[w+1]=0
    # That is: (wire[w] & ~wire[w+1]) should be all zeros
    for w in range(N - 1):
        if np.any(wires[w] & ~wires[w + 1]):
            return False
    return True


def _build_initial_network():
    """Build a valid starting network. Use odd-even transposition (153 comps) 
    but we can also try a smaller known construction."""
    # Odd-even transposition: N phases, each with N//2 comparators
    # Total: N * N//2 = 18 * 9 = 162? Let me recount.
    # Phase p (0-indexed): comparators at (p%2, p%2+2, ...) up to N-1
    # Number of comparators per phase: ceil((N - p%2) / 2)
    # For N=18:
    #   even phases (p%2==0): pairs (0,1),(2,3),...,(16,17) -> 9 comps
    #   odd phases (p%2==1): pairs (1,2),(3,4),...,(15,16) -> 8 comps
    # Total: 9 * 9 + 8 * 9 = 81 + 72 = 153
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _try_remove(comps, idx):
    """Try removing comparator at index idx; return True if still valid."""
    new_comps = comps[:idx] + comps[idx + 1:]
    return _verify_bitwise(new_comps)


def _try_swap(comps, idx1, idx2):
    """Try swapping comparators at idx1 and idx2; return True if still valid."""
    new_comps = list(comps)
    new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
    return _verify_bitwise(new_comps)


def _try_insert(comps, i, j):
    """Try inserting comparator (i,j) at the end; return True if still valid."""
    new_comps = comps + [(i, j)]
    return _verify_bitwise(new_comps)


def construct():
    t_start = time.time()
    rng = random.Random(42)

    # Start with odd-even transposition
    comps = _build_initial_network()
    best_comps = list(comps)
    best_len = len(comps)

    # Verify initial
    if not _verify_bitwise(comps):
        # Should not happen, but safety
        return comps

    # Phase A: Greedy deletion
    # Repeatedly try to remove any comparator until no more can be removed
    changed = True
    while changed and time.time() < DEADLINE - 10:
        changed = False
        for idx in range(len(comps) - 1, -1, -1):  # try from end to start
            if time.time() > DEADLINE - 10:
                break
            if _try_remove(comps, idx):
                comps = comps[:idx] + comps[idx + 1:]
                changed = True
                if len(comps) < best_len:
                    best_len = len(comps)
                    best_comps = list(comps)
                break  # restart scan from end after removal

    # Phase B: Simulated annealing
    if time.time() < DEADLINE - 15:
        T = 2.0  # initial temperature
        cooling = 0.999
        min_T = 0.05
        current_len = len(comps)
        current_comps = list(comps)
        best_sa_len = best_len
        best_sa_comps = list(best_comps)

        # Precompute all possible comparator pairs
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

        while time.time() < DEADLINE - 15 and T > min_T:
            move_type = rng.random()
            if move_type < 0.5:
                # Try removal
                if len(current_comps) > 68:  # don't go below known lower bound
                    idx = rng.randint(0, len(current_comps) - 1)
                    new_comps = current_comps[:idx] + current_comps[idx + 1:]
                    delta = 1  # cost increases by 1 if we remove (we want fewer)
                    # Actually: removing reduces count by 1, so delta_cost = -1 (good)
                    # But if invalid, we reject
                    if _verify_bitwise(new_comps):
                        # Accept (improvement or equal)
                        current_comps = new_comps
                        current_len = len(current_comps)
                        if current_len < best_sa_len:
                            best_sa_len = current_len
                            best_sa_comps = list(current_comps)
                    else:
                        # Reject (removal broke sorting)
                        pass
            elif move_type < 0.8:
                # Try swap
                if len(current_comps) >= 2:
                    idx1 = rng.randint(0, len(current_comps) - 1)
                    idx2 = rng.randint(0, len(current_comps) - 1)
                    if idx1 != idx2:
                        new_comps = list(current_comps)
                        new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                        if _verify_bitwise(new_comps):
                            current_comps = new_comps
                            # length unchanged
                            if current_len < best_sa_len:
                                best_sa_len = current_len
                                best_sa_comps = list(current_comps)
                        # For swap, delta=0, always accept if valid
            else:
                # Try insertion (risky, increases count)
                # Only do this with low probability and only if we're stuck
                if current_len < 120:  # don't bloat too much
                    pair = rng.choice(all_pairs)
                    # Don't insert duplicate
                    if pair not in current_comps:
                        new_comps = current_comps + [pair]
                        if _verify_bitwise(new_comps):
                            delta = 1  # cost increase
                            if rng.random() < np.exp(-delta / T):
                                current_comps = new_comps
                                current_len = len(current_comps)

            T *= cooling

        # Update best
        if best_sa_len < best_len:
            best_len = best_sa_len
            best_comps = list(best_sa_comps)

    # Final greedy deletion pass
    if time.time() < DEADLINE - 5:
        changed = True
        while changed and time.time() < DEADLINE - 5:
            changed = False
            for idx in range(len(best_comps) - 1, -1, -1):
                if time.time() > DEADLINE - 5:
                    break
                if _verify_bitwise(best_comps[:idx] + best_comps[idx + 1:]):
                    best_comps = best_comps[:idx] + best_comps[idx + 1:]
                    changed = True
                    break

    # Final verification
    if not _verify_bitwise(best_comps):
        # Fallback to original
        return _build_initial_network()

    return best_comps
