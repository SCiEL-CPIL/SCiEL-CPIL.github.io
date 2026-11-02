"""Football pool / ternary covering code K_3(8,1).
Greedy seed + randomized local search (removals + targeted swaps) with a deadline guard.
"""
import time
import random
import numpy as np

# The task is fixed to q=3, n=8, radius=1, but we keep the signature general.
def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = t0 + 135.0  # safety margin before the 150 s wall-clock kill
    rng = np.random.default_rng(20240517)

    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct-sum construction (valid, 9*3**(n-4) words for q=3,n>=4)
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tet for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic trivial covering (all words) - always valid
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # ---- specialized q=3, n=8, radius=1 ----
    # Represent every word as an integer in [0, 6561) base-3, length 8.
    pow3 = [1, 3, 9, 27, 81, 243, 729, 2187]
    N = 6561

    # Precompute the 9 neighbors (distance <= 1) for every word.
    # For a word w (digits d0..d7), neighbors: w itself + for each pos i, two variants.
    # Build an array neighbors of shape (N, 9) of int32.
    digits = np.empty((N, 8), dtype=np.int8)
    for i in range(8):
        # digit i = (w // 3**i) % 3
        digits[:, i] = (np.arange(N, dtype=np.int64) // pow3[i]) % 3
    neighbors = np.empty((N, 9), dtype=np.int32)
    neighbors[:, 0] = np.arange(N, dtype=np.int32)
    for i in range(8):
        for delta in (1, 2):
            new_digit = (digits[:, i] + delta) % 3
            neighbors[:, 1 + 2 * i] = (np.arange(N, dtype=np.int32)
                                        + (new_digit - digits[:, i]) * pow3[i])

    # ---- Greedy seed: build a covering, then it's feasible; we will remove. ----
    # Start from a large feasible code, then iteratively remove codewords that are
    # "redundant" (every word they cover is covered by another codeword).
    # We track, for each word, how many codewords cover it (coverage count).
    # A codeword c is removable iff for every word w in ball(c), cover_count[w] >= 2.

    def word_to_list(w):
        return [(w // pow3[i]) % 3 for i in range(8)]

    # Initial code: use the direct-sum construction (9 * 3**4 = 81 words) -- feasible & small.
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    code = []
    for h in tet:
        for tail in itertools.product(range(3), repeat=4):
            code.append(h + list(tail))
    code = np.array(code, dtype=np.int8)

    # Convert code to integer indices.
    def code_to_idx(arr):
        idx = np.zeros(len(arr), dtype=np.int64)
        for i in range(8):
            idx += arr[:, i] * pow3[i]
        return idx.astype(np.int32)

    code_idx = code_to_idx(code)

    # coverage count: how many codewords cover each word.
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code_idx:
        cover_count[neighbors[c]] += 1

    # A codeword is removable iff min over its ball of cover_count >= 2.
    def removable_mask():
        # For each codeword, check min cover_count over its 9 neighbors.
        # code_idx shape (M,). neighbors[code_idx] shape (M,9).
        mins = cover_count[neighbors[code_idx]].min(axis=1)
        return mins >= 2

    # Iteratively remove redundant codewords until none remain.
    def full_sweep_remove():
        progress = True
        while progress:
            progress = False
            rem = removable_mask()
            if rem.any():
                # remove all currently-removable at once (safe? removing several at once
                # can invalidate each other). To be safe, remove one at a time but batch
                # the ones that are mutually independent is complex; do sequential but fast.
                idxs = np.where(rem)[0]
                # Remove them one by one, rechecking minimally.
                for j in idxs:
                    c = code_idx[j]
                    if cover_count[neighbors[c]].min() >= 2:
                        cover_count[neighbors[c]] -= 1
                        code_idx = np.delete(code_idx, j)
                        progress = True
                # after a batch, some may no longer be removable; loop rechecks.
        return code_idx

    # First pass: aggressive removal.
    full_sweep_remove()

    best_code_idx = code_idx.copy()
    best_len = len(code_idx)

    def try_remove_rounds(remaining_time_target):
        nonlocal code_idx, best_code_idx, best_len
        # Randomized removal: repeatedly find a random removable codeword and remove it.
        # To speed up, we maintain the removable set approximately.
        while time.time() < remaining_time_target:
            rem = removable_mask()
            if not rem.any():
                break
            idxs = np.where(rem)[0]
            j = int(rng.choice(idxs))
            c = code_idx[j]
            if cover_count[neighbors[c]].min() >= 2:
                cover_count[neighbors[c]] -= 1
                code_idx = np.delete(code_idx, j)
                if len(code_idx) < best_len:
                    best_len = len(code_idx)
                    best_code_idx = code_idx.copy()

    # Run removal for a chunk of time.
    try_remove_rounds(t0 + 60.0)

    # ---- Local search with targeted swaps to escape local minima ----
    # Objective: minimize |code|. A swap replaces codeword c with c' (c' not in code)
    # such that coverage stays valid. We look for swaps that keep feasibility and
    # enable further removals. We also try "add then remove" moves.

    # Maintain a set of current code indices for O(1) membership.
    def in_code_set(idx_arr):
        s = set(idx_arr.tolist())
        return s

    cur_set = in_code_set(code_idx)

    # Candidate generators: for a codeword c, candidates are words at distance 1
    # from c (the 16 neighbors at distance exactly 1) that are not in the code.
    def candidates_for(c):
        # 16 distance-1 neighbors
        out = []
        for i in range(8):
            for delta in (1, 2):
                nd = ((c // pow3[i]) % 3 + delta) % 3
                cand = c + (nd - (c // pow3[i]) % 3) * pow3[i]
                if cand not in cur_set:
                    out.append(cand)
        return out

    # Try a single swap: replace c with c'. Feasible if after the swap, every word
    # that was covered only by c is still covered by c' (or others).
    # Equivalently: for the ball of c, the words with cover_count==1 must be in ball(c').
    def swap_feasible(c, cp):
        # words uniquely covered by c:
        ball_c = neighbors[c]
        ball_cp = neighbors[cp]
        # words in ball_c with cover_count==1 must be in ball_cp
        uniq = ball_c[cover_count[ball_c] == 1]
        if len(uniq) == 0:
            return True
        # check all uniq are in ball_cp
        return np.all(np.isin(uniq, ball_cp))

    def do_swap(c, cp):
        nonlocal code_idx, cur_set, best_code_idx, best_len
        # remove c
        cover_count[neighbors[c]] -= 1
        # add cp
        cover_count[neighbors[cp]] += 1
        # update code_idx
        pos = np.where(code_idx == c)[0][0]
        code_idx[pos] = cp
        cur_set.discard(c)
        cur_set.add(cp)

    def undo_swap(c, cp):
        nonlocal cur_set
        cover_count[neighbors[cp]] -= 1
        cover_count[neighbors[c]] += 1
        pos = np.where(code_idx == cp)[0][0]
        code_idx[pos] = c
        cur_set.discard(cp)
        cur_set.add(c)

    # Local search loop: alternate removals and swaps.
    swap_budget_end = t0 + 120.0
    while time.time() < swap_budget_end:
        # 1) Remove any redundant codewords.
        rem = removable_mask()
        if rem.any():
            idxs = np.where(rem)[0]
            for j in idxs:
                c = code_idx[j]
                if cover_count[neighbors[c]].min() >= 2:
                    cover_count[neighbors[c]] -= 1
                    code_idx = np.delete(code_idx, j)
                    cur_set.discard(c)
            if len(code_idx) < best_len:
                best_len = len(code_idx)
                best_code_idx = code_idx.copy()
            continue

        # 2) Try targeted swaps. Pick a random codeword, try its candidates.
        if len(code_idx) == 0:
            break
        c = int(rng.choice(code_idx))
        cands = candidates_for(c)
        if not cands:
            continue
        # shuffle candidates
        rng.shuffle(cands)
        swapped = False
        for cp in cands:
            if swap_feasible(c, cp):
                # Perform swap, then try to remove redundancies it enables.
                do_swap(c, cp)
                # try removals
                for _ in range(3):
                    rem2 = removable_mask()
                    if not rem2.any():
                        break
                    idxs2 = np.where(rem2)[0]
                    for j in idxs2:
                        cc = code_idx[j]
                        if cover_count[neighbors[cc]].min() >= 2:
                            cover_count[neighbors[cc]] -= 1
                            code_idx = np.delete(code_idx, j)
                            cur_set.discard(cc)
                if len(code_idx) < best_len:
                    best_len = len(code_idx)
                    best_code_idx = code_idx.copy()
                swapped = True
                break
        # (we keep the swap even if no improvement; it explores the space)

    # Final polish: one more full removal sweep.
    full_sweep_remove()
    if len(code_idx) < best_len:
        best_len = len(code_idx)
        best_code_idx = code_idx.copy()

    # Build the output list of codewords from best_code_idx.
    result = [word_to_list(int(w)) for w in best_code_idx]
    return result
