"""Structural change: LP-guided greedy + iterative deletion + WalkSAT local search.
Combines the LP relaxation duals from the inspiration with the WalkSAT machinery
of the current program. The LP duals guide a weighted greedy initialisation,
which is then refined by deletion and stochastic local search."""
import time
import random
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets and map each to an index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets (candidate blocks) and the t-subsets they cover
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # For each t-subset, list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # Convert to numpy arrays for speed
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # ---------------- LP relaxation to get dual weights ----------------
    duals = None
    try:
        if time.time() < deadline - 60.0:
            rows = []
            cols = []
            data = []
            for j in range(n_k):
                for i in block_covers_np[j]:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)

            M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

            c = np.ones(n_k)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_k

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success and hasattr(res, 'ineqlin') and res.ineqlin is not None:
                duals = np.abs(res.ineqlin.marginals)
                if np.sum(duals) == 0:
                    duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t, dtype=np.float64)

    # ---------------- Weighted greedy initialisation (multiple restarts) ----------------
    best_selected = None
    best_size = float('inf')

    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        # Compute block weights from duals
        block_weights = np.zeros(n_k, dtype=np.float64)
        for j in range(n_k):
            if len(block_covers_np[j]) > 0:
                block_weights[j] = np.sum(duals[block_covers_np[j]])

        # Add noise for diversification
        if restart > 0:
            block_weights += _RNG.uniform(0, 0.05, n_k)

        uncovered = np.ones(n_t, dtype=bool)
        uncovered_count = n_t
        selected = set()

        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                break

            # Find an uncovered t-subset
            target = None
            for i in range(n_t):
                if uncovered[i]:
                    target = i
                    break
            if target is None:
                break

            # Choose block with max weight among those covering target
            best_block = None
            best_w = -1
            for bi in t_to_blocks_np[target]:
                if block_weights[bi] > best_w:
                    best_w = block_weights[bi]
                    best_block = bi

            if best_block is None:
                break

            selected.add(best_block)
            for ti in block_covers_np[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1

        # Ensure full coverage
        while uncovered_count > 0:
            if time.time() > deadline - 2.0:
                break
            target = None
            for i in range(n_t):
                if uncovered[i]:
                    target = i
                    break
            if target is None:
                break
            for bi in t_to_blocks_np[target]:
                if bi not in selected:
                    selected.add(bi)
                    for ti in block_covers_np[bi]:
                        if uncovered[ti]:
                            uncovered[ti] = False
                            uncovered_count -= 1
                    break

        if uncovered_count == 0 and len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)

    if best_selected is None:
        # Fallback: pure greedy
        uncovered = np.ones(n_t, dtype=bool)
        uncovered_count = n_t
        best_selected = set()
        while uncovered_count > 0:
            target = None
            for i in range(n_t):
                if uncovered[i]:
                    target = i
                    break
            if target is None:
                break
            best_block = None
            best_gain = -1
            for bi in t_to_blocks_np[target]:
                gain = sum(1 for ti in block_covers_np[bi] if uncovered[ti])
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            best_selected.add(best_block)
            for ti in block_covers_np[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1

    # ---------------- Deletion reduction ----------------
    selected = best_selected
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    for bi in list(selected):
        if time.time() > deadline - 5.0:
            break
        cov = block_covers_np[bi]
        # Check if any t-subset would become uncovered
        if np.any(cover_count[cov] == 1):
            continue
        # Safe to remove
        cover_count[cov] -= 1
        selected.discard(bi)

    # Repeat deletion until no more changes
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for bi in list(selected):
            if time.time() > deadline - 5.0:
                break
            cov = block_covers_np[bi]
            if np.any(cover_count[cov] == 1):
                continue
            cover_count[cov] -= 1
            selected.discard(bi)
            changed = True

    # ---------------- WalkSAT-style local search ----------------
    x = np.zeros(n_k, dtype=bool)
    for bi in selected:
        x[bi] = True

    clause_sat_count = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat_count[block_covers_np[bi]] += 1

    unsat_clauses = np.where(clause_sat_count == 0)[0]

    best_x = x.copy()
    best_cost = int(np.sum(x)) + 1000 * len(unsat_clauses)

    def compute_cost():
        return int(np.sum(x)) + 1000 * int(np.sum(clause_sat_count == 0))

    def update_best():
        nonlocal best_x, best_cost
        cost = compute_cost()
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

    update_best()

    max_steps = 150000
    for step in range(max_steps):
        if time.time() > deadline - 3.0:
            break

        n_unsat = int(np.sum(clause_sat_count == 0))

        if n_unsat == 0:
            # Try to remove a true variable
            true_vars = np.where(x)[0]
            if len(true_vars) > 0:
                bi = _RNG.choice(true_vars.tolist())
                x[bi] = False
                cov = block_covers_np[bi]
                clause_sat_count[cov] -= 1
                if np.any(clause_sat_count[cov] == 0):
                    # Restore
                    x[bi] = True
                    clause_sat_count[cov] += 1
                else:
                    update_best()
                    continue
            # Random flip to escape local optimum
            if _RNG.random() < 0.15:
                bi = _RNG.randrange(n_k)
                x[bi] = not x[bi]
                cov = block_covers_np[bi]
                clause_sat_count[cov] += 1 if x[bi] else -1
                update_best()
            else:
                # Restart from best
                x = best_x.copy()
                clause_sat_count = np.zeros(n_t, dtype=np.int32)
                for bi in range(n_k):
                    if x[bi]:
                        clause_sat_count[block_covers_np[bi]] += 1
                update_best()
        else:
            # Pick a random unsat clause and a random block covering it
            unsat_list = unsat_clauses.tolist()
            if len(unsat_list) == 0:
                unsat_list = np.where(clause_sat_count == 0)[0].tolist()
            if len(unsat_list) == 0:
                continue
            ti = _RNG.choice(unsat_list)
            candidates = t_to_blocks_np[ti].tolist()
            if not candidates:
                continue
            bi = _RNG.choice(candidates)
            x[bi] = not x[bi]
            cov = block_covers_np[bi]
            clause_sat_count[cov] += 1 if x[bi] else -1
            update_best()

    # Use best found
    final_selected = set(int(bi) for bi in np.where(best_x)[0])

    # Final deletion pass
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in final_selected:
        cover_count[block_covers_np[bi]] += 1

    for bi in list(final_selected):
        if time.time() > deadline - 1.0:
            break
        cov = block_covers_np[bi]
        if np.any(cover_count[cov] == 1):
            continue
        cover_count[cov] -= 1
        final_selected.discard(bi)

    # Build output blocks
    blocks = [list(k_subs[bi]) for bi in sorted(final_selected)]

    if len(blocks) > len(t_subs):
        blocks = blocks[:len(t_subs)]

    return blocks
