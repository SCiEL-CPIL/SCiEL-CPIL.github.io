"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Build all weight-4 vectors (entries in {0,+1,-1}, exactly 4 nonzeros).
    # Norm = 2 for each. Two vectors compatible iff dot <= 0.
    #
    # Represent each vector by its 4 support positions and 4 signs.
    # Precompute compatibility as bitsets for fast independent-set search.

    all_supports = list(itertools.combinations(range(D), 4))
    all_signs = list(itertools.product([1, -1], repeat=4))
    n_signs = len(all_signs)  # 16

    # List of all vectors as (support, sign_tuple)
    vectors = []
    for s in all_supports:
        for sig in all_signs:
            vectors.append((s, sig))
    N = len(vectors)  # 330 * 16 = 5280

    # Precompute sign-dot for each pair of sign tuples (16x16), keyed by overlap pattern.
    # Actually dot depends on which positions overlap and the signs there.
    # Build a matrix compat[a][b] = 1 if vectors a,b compatible (dot<=0) or a==b.

    # For speed, precompute for each vector a list of its (pos, sign) entries.
    entries = []
    for s, sig in vectors:
        entries.append([(s[i], sig[i]) for i in range(4)])

    # Precompute compatibility bitsets using numpy.
    # Build a matrix of dot products.
    # Represent each vector as a length-D array of int.
    mat = np.zeros((N, D), dtype=np.int8)
    for idx, (s, sig) in enumerate(vectors):
        for i in range(4):
            mat[idx, s[i]] = sig[i]

    # Dot products: M = mat @ mat.T
    M = mat @ mat.T  # (N, N) int
    # Compatible if M <= 0 (including self where M=4, but self always allowed)
    # Build adjacency of INCOMPATIBLE pairs: M > 0
    # We want independent set in incompatibility graph.

    # For memory, N=5280, M is 5280x5280 int64 ~ 224 MB. Acceptable.
    # incompatibility mask
    incompat = (M > 0).astype(np.uint8)
    np.fill_diagonal(incompat, 0)

    # For each node, bitset of incompatible neighbors.
    # Use Python ints as bitsets for fast operations.
    Nbits = N
    # Build list of neighbor bitsets.
    # incompat is (N,N) uint8. Convert rows to int bitsets.
    # For each row, we need an int with bit j set if incompat[i,j].
    # Use numpy: treat row as bytes? N=5280 bits = 660 bytes per row.
    # We'll build using: for each i, neighbors = incompat[i].astype(np.uint64) then to int.
    # Simpler: build via loop using int from bytes.
    # incompat[i] is uint8 array of length N. Pack into int.
    neighbor_bits = []
    # Precompute a lookup: convert a uint8 array of length N to int.
    # Use: int.from_bytes(np.packbits(arr).tobytes(), 'little')
    for i in range(N):
        row = incompat[i]
        packed = np.packbits(row)  # length ceil(N/8)
        neighbor_bits.append(int.from_bytes(packed.tobytes(), 'little'))

    # Greedy independent set: repeatedly pick node with fewest remaining incompatible neighbors.
    def greedy_is():
        remaining = set(range(N))
        chosen = []
        # Use a list for remaining for speed; maintain degrees.
        rem = list(range(N))
        # degree within remaining
        deg = [bin(neighbor_bits[i] & 0).count('0') for i in rem]  # placeholder, recompute
        # Better: compute degree as popcount of neighbor_bits[i] & mask_remaining
        # We'll maintain a bitset of remaining.
        rem_mask = (1 << N) - 1
        # degrees
        deg = [bin(neighbor_bits[i] & rem_mask).count('1') for i in rem]
        # Use a simple O(N^2)-ish greedy with periodic recompute.
        # For N=5280 this is heavy; do a faster greedy: sort by degree, pick min, remove.
        # We'll do: while rem: pick i in rem with min deg; add to chosen; for each j in rem incompatible with i, remove j and update deg.
        # Maintain rem as a set and a dict of degrees.
        rem_set = set(rem)
        deg_dict = {i: bin(neighbor_bits[i]).count('1') for i in rem}
        # Actually degree within rem = popcount(neighbor_bits[i] & rem_mask).
        # Recompute lazily.
        # Use a heap-like: just scan each time (N^2 = 28M, ok once).
        chosen = []
        # We'll do a linear-time-per-iteration greedy.
        while rem_set:
            # find min degree in rem_set
            best = None
            bestd = None
            for i in rem_set:
                d = bin(neighbor_bits[i] & rem_mask).count('1')
                if bestd is None or d < bestd:
                    bestd = d
                    best = i
            chosen.append(best)
            # remove best and its incompatible neighbors from rem_set
            rem_set.discard(best)
            nb = neighbor_bits[best]
            # for each j in rem_set incompatible with best: remove
            # iterate over set bits of nb & rem_mask
            m = nb
            to_remove = []
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                m ^= lsb
                if j in rem_set:
                    to_remove.append(j)
            for j in to_remove:
                rem_set.discard(j)
            # update rem_mask
            rem_mask = 0
            for j in rem_set:
                rem_mask |= (1 << j)
        return chosen

    # That greedy is O(N^2) per run with big-int ops; may be slow. 
    # Do a faster greedy: random ordering + first-fit style.
    def greedy_fast():
        order = list(range(N))
        random.shuffle(order)
        chosen = []
        chosen_mask = 0
        for i in order:
            # check if i incompatible with any chosen: (neighbor_bits[i] & chosen_mask) == 0
            if (neighbor_bits[i] & chosen_mask) == 0:
                chosen.append(i)
                chosen_mask |= (1 << i)
        return chosen

    def local_search(start):
        # start: list of chosen indices
        chosen = set(start)
        chosen_mask = 0
        for i in chosen:
            chosen_mask |= (1 << i)
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            # Try to add any node not in chosen that's compatible with all chosen.
            # Then try swaps: remove one, add two? Keep simple: add-only + 1-for-1 swap.
            # Add-only pass:
            added = False
            for i in range(N):
                if i in chosen:
                    continue
                if (neighbor_bits[i] & chosen_mask) == 0:
                    chosen.add(i)
                    chosen_mask |= (1 << i)
                    added = True
                    improved = True
            if added:
                continue
            # 1-for-1 swap: for each i in chosen, for each j not in chosen,
            # if removing i lets us add j (and j incompatible only with i among chosen),
            # that's neutral size; skip. Try 1-for-2: remove i, add two nodes that are
            # incompatible with each other? No, they must be mutually compatible and
            # compatible with chosen\{i}.
            # Do: for each i in chosen, compute the set of nodes compatible with chosen\{i}.
            # Find max independent set in that subgraph? Too heavy. 
            # Simpler: try adding pairs.
            base_mask = 0
            # We'll attempt: for each i, temporarily remove, then greedily add up to 2.
            if len(chosen) < 5000:  # only if room
                for i in list(chosen):
                    if time.time() > deadline - 5:
                        break
                    new_mask = chosen_mask & ~(1 << i)
                    # find two nodes a,b not in chosen, compatible with new_mask and each other
                    cand = []
                    for j in range(N):
                        if j in chosen:
                            continue
                        if (neighbor_bits[j] & new_mask) == 0:
                            cand.append(j)
                    if len(cand) < 2:
                        continue
                    # find a compatible pair in cand
                    found = False
                    for idx_a in range(len(cand)):
                        a = cand[idx_a]
                        for idx_b in range(idx_a + 1, len(cand)):
                            b = cand[idx_b]
                            if (neighbor_bits[a] & (1 << b)) == 0:
                                chosen.discard(i)
                                chosen.add(a)
                                chosen.add(b)
                                chosen_mask = new_mask | (1 << a) | (1 << b)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        return list(chosen)

    # Run several random-order greedy + local search, keep best.
    best = []
    n_trials = 0
    while time.time() < deadline - 10:
        n_trials += 1
        g = greedy_fast()
        ls = local_search(g)
        if len(ls) > len(best):
            best = ls
            # Try to improve best further
            best2 = local_search(best)
            if len(best2) > len(best):
                best = best2
        if n_trials % 5 == 0 and len(best) > 450:
            # good enough, keep going if time
            pass

    # Convert best indices to integer vectors.
    if not best:
        # fallback: basis-ish
        best = [0]
    result = mat[best].astype(np.int64)
    return result

# EVOLVE-BLOCK-END
