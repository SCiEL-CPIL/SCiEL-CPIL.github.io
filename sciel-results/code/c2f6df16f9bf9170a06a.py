"""Structural change: Max-SAT style stochastic local search (WalkSAT) over block-selection
variables with clause = t-subset coverage. We minimise number of selected blocks while
satisfying all coverage clauses. We combine with a greedy initialisation and a
deletion-reduction pass. Explicit deadline guard keeps us under the wall-clock limit."""
import time
import random
import itertools
from collections import defaultdict

# Deterministic seed for reproducibility of stochastic choices
_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s

    # Precompute all t-subsets and map each to an index for fast lookup
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets (candidate blocks) and the t-subsets they cover
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    # For each k-subset, list of t-subset indices it covers
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

    # ---------------- Greedy initial solution ----------------
    uncovered = [True] * n_t
    uncovered_count = n_t
    selected = set()

    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break
        # Pick an uncovered t-subset (first one for determinism)
        target = None
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target is None:
            break
        # Choose the block covering the most uncovered t-subsets among candidates
        best_block = None
        best_gain = -1
        for bi in t_to_blocks[target]:
            gain = 0
            for ti in block_covers[bi]:
                if uncovered[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block is None:
            # Should not happen
            break
        selected.add(best_block)
        for ti in block_covers[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                uncovered_count -= 1

    # Ensure we have a valid covering (if time ran out during greedy, finish it)
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
        best_block = None
        best_gain = -1
        for bi in t_to_blocks[target]:
            gain = 0
            for ti in block_covers[bi]:
                if uncovered[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block is None:
            break
        selected.add(best_block)
        for ti in block_covers[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                uncovered_count -= 1

    # If still not covered (shouldn't happen), add brute-force blocks
    while uncovered_count > 0:
        target = None
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target is None:
            break
        # Pick any block that covers it
        for bi in t_to_blocks[target]:
            if bi not in selected:
                selected.add(bi)
                for ti in block_covers[bi]:
                    if uncovered[ti]:
                        uncovered[ti] = False
                        uncovered_count -= 1
                break

    # ---------------- Deletion reduction ----------------
    # Try to remove each selected block if coverage is still satisfied
    def is_covered(selected_set):
        # Build coverage count array
        cover_count = [0] * n_t
        for bi in selected_set:
            for ti in block_covers[bi]:
                cover_count[ti] += 1
        return all(c > 0 for c in cover_count)

    # We'll maintain cover_count incrementally for speed
    cover_count = [0] * n_t
    for bi in selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1

    def try_remove(bi):
        nonlocal cover_count
        for ti in block_covers[bi]:
            cover_count[ti] -= 1
        # Check if any went to 0
        for ti in block_covers[bi]:
            if cover_count[ti] < 0:
                break
        # Actually check properly
        ok = True
        for ti in block_covers[bi]:
            if cover_count[ti] == 0:
                ok = False
                break
        if ok:
            return True
        # Restore
        for ti in block_covers[bi]:
            cover_count[ti] += 1
        return False

    for bi in list(selected):
        if time.time() > deadline - 5.0:
            break
        if try_remove(bi):
            selected.discard(bi)

    # ---------------- WalkSAT-style local search ----------------
    # State: boolean array of length n_k, True = selected
    x = [bi in selected for bi in range(n_k)]
    # For each clause (t-subset), track which literals satisfy it
    # clause_satisfiers[ti] = list of bi in x that are True and cover ti

    clause_sat_count = [0] * n_t
    for ti in range(n_t):
        clause_sat_count[ti] = sum(1 for bi in t_to_blocks[ti] if x[bi])

    unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]

    def num_unsat():
        return len(unsat_clauses)

    def num_true():
        return sum(1 for xi in x if xi)

    best_x = x[:]
    best_cost = num_true() + 1000 * num_unsat()  # heavy penalty for unsat

    def update_best():
        nonlocal best_x, best_cost
        cost = num_true() + 1000 * num_unsat()
        if cost < best_cost:
            best_cost = cost
            best_x = x[:]

    update_best()

    # WalkSAT loop
    p_flip_unsat = 0.5  # probability to flip a variable in an unsatisfied clause
    max_steps = 200000

    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        if num_unsat() == 0:
            # Try to remove a true variable that doesn't break coverage
            true_vars = [bi for bi in range(n_k) if x[bi]]
            if true_vars:
                bi = _RNG.choice(true_vars)
                x[bi] = False
                for ti in block_covers[bi]:
                    clause_sat_count[ti] -= 1
                new_unsat = [ti for ti in block_covers[bi] if clause_sat_count[ti] == 0]
                if not new_unsat:
                    unsat_clauses = []
                    update_best()
                    continue
                else:
                    # Restore
                    x[bi] = True
                    for ti in block_covers[bi]:
                        clause_sat_count[ti] += 1
                    unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
                    update_best()
                    continue
            # No true vars to remove, try flipping a false var to true to cover unsat
            if unsat_clauses:
                ti = _RNG.choice(unsat_clauses)
                candidates = [bi for bi in t_to_blocks[ti] if not x[bi]]
                if candidates:
                    bi = _RNG.choice(candidates)
                    x[bi] = True
                    for ti2 in block_covers[bi]:
                        clause_sat_count[ti2] += 1
                    unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
                    update_best()
                    continue
            # If no unsat and no removable, we're at a local optimum for this state
            # Try a random flip to escape
            if _RNG.random() < 0.1:
                bi = _RNG.randrange(n_k)
                x[bi] = not x[bi]
                for ti in block_covers[bi]:
                    clause_sat_count[ti] += 1 if x[bi] else -1
                unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
                update_best()
                continue
            # Otherwise, restart with best
            x = best_x[:]
            clause_sat_count = [0] * n_t
            for bi in range(n_k):
                if x[bi]:
                    for ti in block_covers[bi]:
                        clause_sat_count[ti] += 1
            unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
            update_best()
            continue

        # There are unsatisfied clauses
        # With probability p_flip_unsat, pick a var from a random unsat clause
        # Otherwise, pick a random var that appears in some unsat clause
        if _RNG.random() < p_flip_unsat:
            ti = _RNG.choice(unsat_clauses)
            candidates = t_to_blocks[ti]
            if candidates:
                bi = _RNG.choice(candidates)
                # Flip
                x[bi] = not x[bi]
                for ti2 in block_covers[bi]:
                    clause_sat_count[ti2] += 1 if x[bi] else -1
                unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
                update_best()
        else:
            # Pick a random unsat clause, then a random var in it
            ti = _RNG.choice(unsat_clauses)
            candidates = t_to_blocks[ti]
            if candidates:
                bi = _RNG.choice(candidates)
                x[bi] = not x[bi]
                for ti2 in block_covers[bi]:
                    clause_sat_count[ti2] += 1 if x[bi] else -1
                unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
                update_best()

    # Use best found
    final_selected = set(bi for bi in range(n_k) if best_x[bi])

    # Final deletion reduction on the best solution
    cover_count = [0] * n_t
    for bi in final_selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1

    for bi in list(final_selected):
        if time.time() > deadline - 2.0:
            break
        for ti in block_covers[bi]:
            cover_count[ti] -= 1
        ok = True
        for ti in block_covers[bi]:
            if cover_count[ti] == 0:
                ok = False
                break
        if ok:
            final_selected.discard(bi)
        else:
            for ti in block_covers[bi]:
                cover_count[ti] += 1

    # Build output blocks
    blocks = [list(k_subs[bi]) for bi in sorted(final_selected)]

    # Validate
    if len(blocks) > len(t_subs):
        # Truncate if somehow too many (shouldn't happen)
        blocks = blocks[:len(t_subs)]

    return blocks
