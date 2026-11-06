"""Edge-colourings of K16 in which every K4 sees five colours."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """Return 120 integer colours for edges of K16 (lex order i<j).
    Every K4 must span at least 5 distinct colours. Minimise number of colours used."""
    t0 = time.time()
    deadline = t0 + 112.0  # safety margin before 120s kill

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s and their edge indices
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4_edges.append([edge_idx[x] for x in e])
    num_k4 = len(k4s)  # 1820

    # For each edge, the list of (k4_index, position_within_k4) that contain it
    edge_to_k4s = [[] for _ in range(m)]
    for ki, ke in enumerate(k4_edges):
        for pos, ei in enumerate(ke):
            edge_to_k4s[ei].append((ki, pos))

    # Numpy arrays for fast K4 colour counting
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)

    # Initial solution: (i+j) mod 17 -> valid 17-colour solution
    sol = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)

    def colors_used_count(arr):
        return len(np.unique(arr))

    def is_valid(arr):
        # arr: numpy int array length m
        # For each K4 (6 edges), count distinct colours; require >= 5
        c = arr[k4_edges_arr]  # shape (1820, 6)
        # distinct count per row
        # Use: number of distinct = 6 - number of "duplicates" is not direct;
        # compute via sorting each row and counting changes
        cs = np.sort(c, axis=1)
        # distinct = 1 + sum(cs[:,1:] != cs[:,:-1])
        distinct = 1 + np.sum(cs[:, 1:] != cs[:, :-1], axis=1)
        return bool(np.all(distinct >= 5))

    # Fast validity check after a single edge recolor: only affected K4s matter
    # affected_k4s: list of k4 indices; we recompute distinct only for those rows
    def check_edge_recolor(arr, ei, new_c, affected_k4_indices):
        # arr is the working numpy array; we assume arr[ei] already set to new_c
        if len(affected_k4_indices) == 0:
            return True
        rows = k4_edges_arr[affected_k4_indices]  # shape (k, 6)
        c = arr[rows]
        cs = np.sort(c, axis=1)
        distinct = 1 + np.sum(cs[:, 1:] != cs[:, :-1], axis=1)
        return bool(np.all(distinct >= 5))

    # Precompute affected k4 index arrays per edge (as numpy int arrays for fast indexing)
    edge_affected_arr = [np.array([ki for (ki, pos) in lst], dtype=np.int32) for lst in edge_to_k4s]

    def recolor_valid(arr, ei, new_c):
        """Temporarily set arr[ei]=new_c, check validity of affected K4s, restore."""
        old = arr[ei]
        arr[ei] = new_c
        ok = check_edge_recolor(arr, ei, new_c, edge_affected_arr[ei])
        arr[ei] = old
        return ok

    # Colour-frequency driven local search to minimise number of distinct colours used.
    # Strategy: repeatedly find the most-used colour, and try to recolor some of its
    # edges to a less-used colour while preserving validity.

    def frequency_search(arr, time_limit, max_rounds=200000):
        arr = arr.copy()
        cur_used = colors_used_count(arr)
        rng = random.Random(12345)
        rounds = 0
        while time.time() < time_limit and rounds < max_rounds:
            rounds += 1
            freq = np.bincount(arr, minlength=cur_used + 1)
            # colours actually present
            present = [c for c in range(len(freq)) if freq[c] > 0]
            if len(present) <= 1:
                break
            # most used colour
            max_c = max(present, key=lambda c: freq[c])
            if freq[max_c] <= 1:
                # all colours used at most once; can't reduce further by this method
                break
            # candidate target colours: less-used colours (prefer lower count, then lower index)
            targets = sorted([c for c in present if c != max_c], key=lambda c: (freq[c], c))
            if not targets:
                break
            # edges holding max_c
            max_edges = [ei for ei in range(m) if arr[ei] == max_c]
            rng.shuffle(max_edges)
            progress = False
            for ei in max_edges:
                if time.time() >= time_limit:
                    break
                # try targets in order
                for new_c in targets:
                    if recolor_valid(arr, ei, new_c):
                        arr[ei] = new_c
                        progress = True
                        break
                if progress:
                    # recompute and continue
                    break
            if not progress:
                # stuck: try a random perturbation to escape
                if time.time() >= time_limit:
                    break
                # random perturbation: recolor a random edge to a random used colour
                # (accept only if valid, to keep a valid solution)
                for _ in range(200):
                    ei = rng.randrange(m)
                    old = arr[ei]
                    new_c = rng.choice(present)
                    if new_c == old:
                        continue
                    if recolor_valid(arr, ei, new_c):
                        arr[ei] = new_c
                        break
            # update used count
            new_used = colors_used_count(arr)
            if new_used < cur_used:
                cur_used = new_used
        return arr, cur_used

    best_sol = sol.copy()
    best_used = colors_used_count(best_sol)

    # Run frequency search
    if time.time() < deadline:
        t_limit = deadline - 2.0
        s2, used2 = frequency_search(best_sol, t_limit)
        if is_valid(s2) and used2 < best_used:
            best_sol = s2
            best_used = used2

    # Final high-precision polish: try to eliminate the single most-used colour
    # by recoloring all of its edges one by one to least-used colours.
    if time.time() < deadline:
        t_limit = deadline - 1.0
        arr = best_sol.copy()
        for _ in range(50):
            if time.time() >= t_limit:
                break
            freq = np.bincount(arr, minlength=best_used + 1)
            present = [c for c in range(len(freq)) if freq[c] > 0]
            if len(present) <= 1:
                break
            max_c = max(present, key=lambda c: freq[c])
            if freq[max_c] <= 1:
                break
            targets = sorted([c for c in present if c != max_c], key=lambda c: (freq[c], c))
            max_edges = [ei for ei in range(m) if arr[ei] == max_c]
            random.shuffle(max_edges)
            improved = False
            for ei in max_edges:
                if time.time() >= t_limit:
                    break
                for new_c in targets:
                    if recolor_valid(arr, ei, new_c):
                        arr[ei] = new_c
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if is_valid(arr):
            u = colors_used_count(arr)
            if u < best_used:
                best_sol = arr
                best_used = u

    # Final validation (defensive)
    if not is_valid(best_sol):
        # Fallback to the guaranteed valid construction
        best_sol = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)

    # Return as a plain Python list of ints in lex order
    return [int(x) for x in best_sol]

# EVOLVE-BLOCK-END
