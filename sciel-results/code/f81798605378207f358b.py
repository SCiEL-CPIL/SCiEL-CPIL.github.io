"""Rewritten program: greedy independent set on the weight-4 conflict graph with
Kempe-chain swap polish, plus auxiliary weight-2/weight-1 vectors."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def _build_w4_data():
    """Generate all weight-4 vectors and the conflict graph (adjacency as sets)."""
    supports = list(itertools.combinations(range(D), 4))
    w4_vectors = []
    support_to_indices = defaultdict(list)
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            idx = len(w4_vectors)
            w4_vectors.append((sup, signs))
            support_to_indices[sup].append(idx)
    n = len(w4_vectors)

    # Conflict graph: for weight-4 vectors, conflict iff |intersection|>=3 and
    # all overlapping positions have the same sign.
    conflicts = [set() for _ in range(n)]

    support_list = list(support_to_indices.keys())
    n_sup = len(support_list)

    # Precompute intersections of size 3
    for i in range(n_sup):
        S1 = support_list[i]
        for j in range(i + 1, n_sup):
            S2 = support_list[j]
            inter = set(S1) & set(S2)
            if len(inter) != 3:
                continue
            inter_sorted = sorted(inter)
            pos_in_S1 = [S1.index(p) for p in inter_sorted]
            pos_in_S2 = [S2.index(p) for p in inter_sorted]
            idx1_list = support_to_indices[S1]
            idx2_list = support_to_indices[S2]
            for idx1 in idx1_list:
                sig1 = w4_vectors[idx1][1]
                for idx2 in idx2_list:
                    sig2 = w4_vectors[idx2][1]
                    all_same = all(sig1[p1] == sig2[p2] for p1, p2 in zip(pos_in_S1, pos_in_S2))
                    if all_same:
                        conflicts[idx1].add(idx2)
                        conflicts[idx2].add(idx1)
    return w4_vectors, conflicts, n

def _greedy_independent_set(conflicts, n, deadline, rng):
    """Greedy minimum-degree independent set with randomized tie-breaking."""
    degrees = np.array([len(c) for c in conflicts], dtype=np.int32)
    remaining = np.ones(n, dtype=bool)
    selected = []

    while remaining.any():
        if time.time() > deadline:
            break
        rem_indices = np.where(remaining)[0]
        if len(rem_indices) == 0:
            break
        rem_degrees = degrees[rem_indices]
        min_deg = np.min(rem_degrees)
        min_mask = rem_degrees == min_deg
        min_indices_local = np.where(min_mask)[0]
        chosen_local = rng.integers(0, len(min_indices_local))
        chosen_global = rem_indices[chosen_local]
        selected.append(chosen_global)

        to_remove = set(conflicts[chosen_global])
        to_remove.add(chosen_global)
        for r in to_remove:
            if remaining[r]:
                remaining[r] = False
                for nb in conflicts[r]:
                    if remaining[nb]:
                        degrees[nb] -= 1
    return selected

def _kempe_polish(conflicts, n, selected_set, selected_list, deadline, rng, max_rounds=200):
    """Try Kempe-chain swaps: remove a small independent set from the complement
    and add a larger one, if the net gain is positive."""
    selected_set = set(selected_set)
    selected_list = list(selected_set)
    n_selected = len(selected_list)

    # Build a mapping: for each unselected vertex, its degree into selected set (number of neighbors in selected)
    # A vertex can be added if it has no neighbors in selected_set.
    # For swaps: find a small set R of selected vertices to remove, then find a larger set A of
    # vertices that become addable after removing R.

    # Simpler approach: try single-vertex removal + multi-add (Kempe chain of length 1 on selected side)
    # For each selected vertex v, if we remove v, which unselected vertices become addable?
    # An unselected vertex u becomes addable if all its selected neighbors are removed.
    # After removing v, u is addable iff all neighbors of u in selected_set are {v} (i.e., v is the only selected neighbor).

    # Precompute for each unselected vertex, the set of its selected neighbors
    unselected = [i for i in range(n) if i not in selected_set]
    # This is expensive to recompute; do it once per polish call.

    # For efficiency, only try a subset of removals per round
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # Try removing each selected vertex and see if we can add more than 1
        # To keep it fast, only try a random subset
        candidates_to_remove = list(selected_list)
        rng.shuffle(candidates_to_remove)
        max_tries = min(len(candidates_to_remove), 500)
        for v in candidates_to_remove[:max_tries]:
            if time.time() > deadline:
                break
            # Find unselected vertices whose only selected neighbor is v
            # i.e., neighbors in selected_set == {v}
            addable = []
            for u in unselected:
                if u in selected_set:
                    continue
                sel_neighbors = [nb for nb in conflicts[u] if nb in selected_set]
                if sel_neighbors == [v]:
                    addable.append(u)
            if len(addable) > 1:
                # Remove v, add all addable
                selected_set.remove(v)
                selected_list.remove(v)
                for u in addable:
                    selected_set.add(u)
                    selected_list.append(u)
                # Update unselected
                unselected = [i for i in range(n) if i not in selected_set]
                improved = True
                break
    return selected_set, selected_list

def _try_add_aux_vectors(w4_selected_indices, w4_vectors, conflicts, n, deadline):
    """Try to add weight-2 and weight-1 doubled vectors that don't conflict."""
    # Build the set of selected w4 vectors as actual integer vectors
    selected_w4 = [w4_vectors[i] for i in w4_selected_indices]

    # Weight-1 doubled vectors: 22 of them, never conflict with w4
    w1_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            w1_vectors.append(v)

    # Weight-2 vectors: check conflict with selected w4
    # Conflict with w4 iff they share both positions AND have same signs at both
    w4_blocked_w2 = set()
    for sup, sig in selected_w4:
        for a, b in itertools.combinations(range(4), 2):
            pos_a, pos_b = sup[a], sup[b]
            sig_a, sig_b = sig[a], sig[b]
            if pos_a > pos_b:
                pos_a, pos_b = pos_b, pos_a
                sig_a, sig_b = sig_b, sig_a
            w4_blocked_w2.add((pos_a, pos_b, sig_a, sig_b))

    # Weight-2 candidates
    w2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                w2_candidates.append((i, j, si, sj))

    # w2 conflicts with w1 if they share a position with same sign
    # Since we add all w1, no w2 can be added (as analyzed in original)
    # So choose: either all w1 (22) or all compatible w2 (no w1)

    w2_compatible = [(i, j, si, sj) for (i, j, si, sj) in w2_candidates
                     if (i, j, si, sj) not in w4_blocked_w2]

    if len(w2_compatible) >= len(w1_vectors):
        aux = []
        for (i, j, si, sj) in w2_compatible:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            aux.append(v)
        return np.array(aux, dtype=np.int64) if aux else np.empty((0, D), dtype=np.int64)
    else:
        return np.array(w1_vectors, dtype=np.int64)

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build w4 data
    w4_vectors, conflicts, n = _build_w4_data()

    # Greedy independent set with multiple restarts
    rng = np.random.default_rng(42)
    best_selected = []
    best_size = 0

    max_restarts = 300
    for attempt in range(max_restarts):
        if time.time() > deadline - 30:
            break
        selected = _greedy_independent_set(conflicts, n, deadline - 30, rng)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            # Polish with Kempe chains
            sel_set, sel_list = _kempe_polish(conflicts, n, set(selected), selected,
                                               deadline - 20, rng, max_rounds=50)
            if len(sel_list) > best_size:
                best_size = len(sel_list)
                best_selected = sel_list.copy()

    # Build integer vectors for selected w4
    w4_int_vectors = np.zeros((best_size, D), dtype=np.int64)
    for i, idx in enumerate(best_selected):
        sup, sig = w4_vectors[idx]
        for pos, sign in zip(sup, sig):
            w4_int_vectors[i, pos] = sign

    # Add auxiliary vectors
    aux_vectors = _try_add_aux_vectors(best_selected, w4_vectors, conflicts, n, deadline)

    # Combine
    if len(aux_vectors) > 0:
        all_vectors = np.vstack([w4_int_vectors, aux_vectors])
    else:
        all_vectors = w4_int_vectors

    return all_vectors
# EVOLVE-BLOCK-END
