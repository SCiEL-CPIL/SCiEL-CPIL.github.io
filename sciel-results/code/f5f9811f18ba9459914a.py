"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START
def construct():
    """
    Find a size-efficient sorting network on 18 wires using:
    1. A fast bitset-based 0-1 verification (checks all 2^18 Boolean inputs).
    2. Greedy comparator removal with random restarts and simulated-annealing-like
       acceptance of neutral moves to escape local minima.
    3. A deadline guard to return the best network found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s kill

    # ---------- Fast 0-1 verification using bitsets ----------
    # Represent the state of all 2^N inputs as N bitsets (one per wire),
    # where bitset[w][mask] = 1 iff wire w has value 1 for input mask.
    # Initially, for input mask (bitstring), wire w has bit (mask >> w) & 1.
    # We can build these efficiently.

    def _init_bitsets():
        """Return list of N numpy uint64 arrays, each of length TOTAL_INPUTS//64,
        representing the N wires across all 2^N inputs."""
        # Each bitset is an array of 64-bit integers.
        num_chunks = TOTAL_INPUTS // 64
        bitsets = []
        for w in range(N):
            # For wire w, the bit is (mask >> w) & 1.
            # We can build this pattern: for each chunk of 64 masks,
            # the pattern depends on w and the chunk index.
            arr = np.zeros(num_chunks, dtype=np.uint64)
            if w == 0:
                # Alternating 1,0,1,0,... within each 64-block: 0x5555555555555555
                arr[:] = 0x5555555555555555
            elif w == 1:
                # 00110011... pattern: 0x3333333333333333
                arr[:] = 0x3333333333333333
            elif w == 2:
                # 00001111...: 0x0F0F0F0F0F0F0F0F
                arr[:] = 0x0F0F0F0F0F0F0F0F
            elif w == 3:
                # 0000000011111111: 0x00FF00FF00FF00FF
                arr[:] = 0x00FF00FF00FF00FF
            else:
                # For w >= 4, the pattern repeats every 2^w bits.
                # Within a 64-bit chunk, the period is 2^w.
                # We can compute: for each position p in [0,63], bit = (p >> w) & 1 if w < 6
                # Actually, let's just compute directly for simplicity since N=18 is small.
                # But for speed, let's precompute the 64-bit pattern for each w.
                pattern = 0
                for p in range(64):
                    if (p >> w) & 1:
                        pattern |= (1 << p)
                arr[:] = pattern
            bitsets.append(arr)
        return bitsets

    def _apply_comparator(bitsets, i, j):
        """Apply compare-exchange on wires i and j: min to i, max to j."""
        # min(i,j) = bitsets[i] & bitsets[j]
        # max(i,j) = bitsets[i] | bitsets[j]
        new_i = bitsets[i] & bitsets[j]
        new_j = bitsets[i] | bitsets[j]
        bitsets[i] = new_i
        bitsets[j] = new_j

    def _is_sorted(bitsets):
        """Check if the network sorts all inputs: for each input, wire 0 has 1 iff
        all inputs are 1, wire 1 has 1 iff at least 2 inputs are 1, etc.
        Actually, for a sorting network, after sorting, wire k should have value 1
        iff the number of 1s in the input is > k. So wire 0 = all ones, wire N-1 = any one.
        More precisely: wire k is 1 iff popcount(input) > k.
        We can check: wire 0 should be the AND of all initial wires, wire N-1 should be OR.
        Actually, the sorted output for input with m ones is: first (N-m) wires are 0, last m wires are 1.
        So wire k is 1 iff m > k, i.e., iff the sum of input bits > k.
        For 0-1 principle, we just need to check that the output is sorted: wire 0 <= wire 1 <= ... <= wire N-1.
        Since values are 0/1, this means if wire k is 0, all higher wires must also be 0? No.
        Sorted means non-decreasing: wire[0] <= wire[1] <= ... <= wire[N-1].
        For 0-1 values, this means once we see a 1, all subsequent must be 1.
        So wire k is 1 implies wire k+1 is 1, ..., wire N-1 is 1.
        Equivalently: wire[k] & ~wire[k+1] must be 0 for all k.
        """
        for k in range(N - 1):
            if np.any(bitsets[k] & ~bitsets[k + 1]):
                return False
        return True

    def _verify_network(comparators):
        """Verify that the given list of comparators sorts all 2^N inputs."""
        bitsets = _init_bitsets()
        for i, j in comparators:
            _apply_comparator(bitsets, i, j)
        return _is_sorted(bitsets)

    # ---------- Seed: odd-even transposition ----------
    def _odd_even_transposition():
        comps = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                comps.append((i, i + 1))
        return comps

    # ---------- Greedy reduction ----------
    def _greedy_reduce(comparators, rng, time_limit):
        """
        Try to remove comparators one by one. Use random order and try removal.
        If removal keeps the network valid, keep it removed.
        Also try swapping adjacent comparators (if they don't interfere) to explore.
        """
        comps = list(comparators)
        n = len(comps)
        # Randomly shuffle for diversity
        rng.shuffle(comps)

        # Greedy removal pass
        changed = True
        while changed:
            changed = False
            for idx in range(len(comps)):
                if time.time() > time_limit:
                    break
                # Try removing comps[idx]
                if len(comps) <= 1:
                    break
                new_comps = comps[:idx] + comps[idx + 1:]
                if _verify_network(new_comps):
                    comps = new_comps
                    changed = True
                    break  # Restart the loop since indices changed
            if time.time() > time_limit:
                break

        return comps

    # ---------- Simulated annealing on comparator order ----------
    def _anneal(comparators, rng, time_limit):
        """
        Try to reorder comparators to enable further removals.
        """
        comps = list(comparators)
        current_len = len(comps)
        best_comps = list(comps)
        best_len = current_len

        while time.time() < time_limit:
            # Pick two random positions and swap
            if len(comps) < 2:
                break
            i, j = rng.sample(range(len(comps)), 2)
            comps[i], comps[j] = comps[j], comps[i]
            if _verify_network(comps):
                if len(comps) < best_len:
                    best_comps = list(comps)
                    best_len = len(comps)
            else:
                # Revert
                comps[i], comps[j] = comps[j], comps[i]

        return best_comps

    # ---------- Main search loop with restarts ----------
    best_network = _odd_even_transposition()
    best_size = len(best_network)

    rng = random.Random(42)

    while time.time() < deadline:
        # Start from a perturbed version of the best network
        current = list(best_network)

        # Add some random comparators to escape local minima
        num_adds = rng.randint(0, 5)
        for _ in range(num_adds):
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            current.append((i, j))

        # Greedy reduce
        reduced = _greedy_reduce(current, rng, deadline)

        # Try annealing
        annealed = _anneal(reduced, rng, deadline)

        if len(annealed) < best_size:
            best_network = annealed
            best_size = len(annealed)

        # Also try direct greedy on the current best
        direct = _greedy_reduce(best_network, rng, deadline)
        if len(direct) < best_size:
            best_network = direct
            best_size = len(direct)

    return best_network
# EVOLVE-BLOCK-END
