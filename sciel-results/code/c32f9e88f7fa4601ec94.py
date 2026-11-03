"""Kissing number in dimension 11: support-level MIS + vector-level refinement + lower-weight vectors."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build conflict graph at support level: edge if |intersection| == 3
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    support_sets = [set(s) for s in supports]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj_mask[i, j] = 1
                adj_mask[j, i] = 1

    # Precompute adjacency lists
    adj_lists = [np.where(adj_mask[i] == 1)[0] for i in range(n_sup)]

    def greedy_is(seed=0):
        """Greedy independent set with minimum-degree heuristic and random tie-breaking."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)

        while True:
            if time.time() > deadline - 20.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break

            deg = (adj_mask * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n_sup + 1

            min_deg = deg.min()
            candidates = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates) == 0:
                break

            pick = rng.choice(candidates)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[adj_mask[pick] == 1] = 0

        return selected

    def sa_support(S_init, n_steps=50000, seed=0):
        """Simulated annealing at support level with 1-for-k moves."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)

        S = S_init.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()

        T = 2.0
        cooling = 0.9997
        min_T = 0.05

        step = 0
        while step < n_steps and time.time() < deadline - 15.0:
            step += 1
            v = rng.integers(0, n_sup)

            if S[v]:
                if rand.random() < 0.03:
                    S[v] = 0
                    current_size -= 1
                continue

            conflicts = np.where(adj_mask[v] & S)[0]
            n_conf = len(conflicts)

            if n_conf == 0:
                S[v] = 1
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                delta = 1 - n_conf
                if delta >= 0:
                    S[conflicts] = 0
                    S[v] = 1
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    if T > min_T:
                        p = np.exp(delta / T)
                        if rand.random() < p:
                            S[conflicts] = 0
                            S[v] = 1
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()

            T = max(min_T, T * cooling)

        return best_S, best_size

    # Multiple restarts at support level
    best_S = np.zeros(n_sup, dtype=np.uint8)
    best_size = 0

    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        S_init = greedy_is(seed=1000 + i)
        S_sa, sa_size = sa_support(S_init, n_steps=40000, seed=2000 + i)

        if sa_size > best_size:
            best_size = sa_size
            best_S = S_sa.copy()

    # Convert best support set to vectors: all 16 sign patterns per support
    vectors = []
    for idx in range(n_sup):
        if best_S[idx]:
            sup = supports[idx]
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for k, s in enumerate(sup):
                    v[s] = signs[k]
                vectors.append(v)

    existing = np.array(vectors, dtype=np.int32)
    n_existing = len(existing)

    # Now do vector-level refinement: try to swap out individual vectors
    # Build vector-level conflict: two vectors conflict if |dot| > 1
    # We'll work with the existing set and try local moves

    def vector_local_search(existing, n_steps=30000, seed=0):
        """Local search at vector level: try to replace vectors to increase count."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)

        vecs = list(existing)
        n_vecs = len(vecs)
        best_vecs = vecs.copy()
        best_n = n_vecs

        # Precompute norms squared
        norms_sq = np.array([sum(x*x for x in v) for v in vecs], dtype=np.int32)
        max_norm_sq = int(norms_sq.max())

        T = 1.0
        cooling = 0.9998
        min_T = 0.02

        step = 0
        while step < n_steps and time.time() < deadline - 10.0:
            step += 1

            # Pick a random vector to potentially remove
            idx_remove = rng.integers(0, n_vecs)
            v_remove = vecs[idx_remove]

            # Try to find a candidate to add that is compatible with all others
            # Generate random candidate: random support and signs
            sup = list(itertools.combinations(range(D), 4))
            sup_idx = rng.integers(0, n_sup)
            signs = tuple(rng.integers(0, 2, size=4))
            signs = tuple(1 if s == 0 else -1 for s in signs)

            v_add = [0] * D
            for k, s in enumerate(sup[sup_idx]):
                v_add[s] = signs[k]

            # Check if v_add is already in vecs
            if v_add in vecs:
                continue

            # Check compatibility: for all other vectors (except v_remove), |dot| <= 1
            # And for v_remove itself, we're removing it so no constraint
            compatible = True
            for j, vj in enumerate(vecs):
                if j == idx_remove:
                    continue
                dot = sum(a*b for a, b in zip(v_add, vj))
                if abs(dot) > 1:
                    compatible = False
                    break

            if compatible:
                vecs[idx_remove] = v_add
                # Update norms
                norms_sq[idx_remove] = sum(x*x for x in v_add)
                max_norm_sq = max(max_norm_sq, int(norms_sq[idx_remove]))
                # Count didn't change, but we might have improved the set
                # Actually this is a swap, not an increase. Let's try a different approach.
                # Revert and try adding without removing
                vecs[idx_remove] = v_remove
                norms_sq[idx_remove] = sum(x*x for x in v_remove)
            else:
                # Try to add v_add by removing all conflicting vectors
                conflicts = []
                for j, vj in enumerate(vecs):
                    dot = sum(a*b for a, b in zip(v_add, vj))
                    if abs(dot) > 1:
                        conflicts.append(j)

                if len(conflicts) == 0:
                    # No conflicts, just add
                    vecs.append(v_add)
                    n_vecs += 1
                    norms_sq = np.append(norms_sq, sum(x*x for x in v_add))
                    max_norm_sq = max(max_norm_sq, int(norms_sq[-1]))
                    if n_vecs > best_n:
                        best_n = n_vecs
                        best_vecs = vecs.copy()
                elif len(conflicts) == 1:
                    # 1-for-1 swap
                    c_idx = conflicts[0]
                    vecs[c_idx] = v_add
                    norms_sq[c_idx] = sum(x*x for x in v_add)
                    max_norm_sq = max(max_norm_sq, int(norms_sq[c_idx]))
                    # Size unchanged, skip
                elif len(conflicts) <= 3:
                    # 1-for-k move: remove k conflicting, add 1
                    delta = 1 - len(conflicts)
                    if delta >= 0 or (T > min_T and rand.random() < np.exp(delta / T)):
                        for c in sorted(conflicts, reverse=True):
                            vecs.pop(c)
                            norms_sq = np.delete(norms_sq, c)
                        n_vecs -= len(conflicts)
                        vecs.append(v_add)
                        n_vecs += 1
                        norms_sq = np.append(norms_sq, sum(x*x for x in v_add))
                        max_norm_sq = max(max_norm_sq, int(norms_sq[-1]))
                        if n_vecs > best_n:
                            best_n = n_vecs
                            best_vecs = vecs.copy()

            T = max(min_T, T * cooling)

        return best_vecs

    # Do vector-level local search from the support-level result
    existing = vector_local_search(existing, n_steps=20000, seed=42)

    # Now add lower-weight vectors
    def try_add_lower_weight(existing, weight, max_dot_allowed, deadline_margin=5.0):
        """Try to add lower-weight vectors that are compatible with all existing."""
        added = 0
        if weight == 1:
            candidates = []
            for pos in range(D):
                for s in [1, -1]:
                    v = [0] * D
                    v[pos] = s
                    candidates.append(v)
        elif weight == 2:
            candidates = []
            for positions in itertools.combinations(range(D), 2):
                for signs in itertools.product([1, -1], repeat=2):
                    v = [0] * D
                    for pos, s in zip(positions, signs):
                        v[pos] = s
                    candidates.append(v)
        elif weight == 3:
            candidates = []
            for positions in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = [0] * D
                    for pos, s in zip(positions, signs):
                        v[pos] = s
                    candidates.append(v)
        else:
            return existing, 0

        cand_arr = np.array(candidates, dtype=np.int32)
        existing_arr = np.array(existing, dtype=np.int32)

        # Batch check: for each candidate, compute max |dot| with existing
        # dots[i, j] = cand[i] . existing[j]
        dots = cand_arr @ existing_arr.T
        max_abs_dots = np.max(np.abs(dots), axis=1)

        # Candidates with max |dot| <= max_dot_allowed are compatible
        compatible_idx = np.where(max_abs_dots <= max_dot_allowed)[0]

        # Sort by how close they are to being compatible (fewer conflicts first)
        if len(compatible_idx) > 0:
            # Add all compatible ones
            for idx in compatible_idx:
                if time.time() > deadline - deadline_margin:
                    break
                existing.append(candidates[idx])
                added += 1

        # For non-compatible, sort by max_abs_dots and try in order
        non_compat = np.where(max_abs_dots > max_dot_allowed)[0]
        if len(non_compat) > 0:
            non_compat_sorted = non_compat[np.argsort(max_abs_dots[non_compat])]
            existing_arr = np.array(existing, dtype=np.int32)
            for idx in non_compat_sorted:
                if time.time() > deadline - deadline_margin:
                    break
                c = cand_arr[idx]
                dots_c = existing_arr @ c
                if np.all(np.abs(dots_c) <= max_dot_allowed):
                    existing.append(candidates[idx])
                    existing_arr = np.vstack([existing_arr, c.reshape(1, -1)])
                    added += 1

        return existing, added

    # Add weight-3 vectors (max |dot| <= 1)
    existing, _ = try_add_lower_weight(existing, 3, 1, deadline_margin=8.0)

    # Add weight-2 vectors (max |dot| <= 1)
    existing, _ = try_add_lower_weight(existing, 2, 1, deadline_margin=5.0)

    # Add weight-1 vectors (max |dot| <= 0, i.e., dot == 0)
    existing, _ = try_add_lower_weight(existing, 1, 0, deadline_margin=3.0)

    if not existing:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = np.array(existing, dtype=np.int64)
    return result
