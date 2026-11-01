import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate candidate pool:
    # Type 1: two ±1's, rest 0 (norm^2 = 2) -> 220 vectors
    # Type 2: one ±2, rest 0 (norm^2 = 4) -> 22 vectors
    # Type 3: four ±1's, rest 0 (norm^2 = 4) -> 5280 vectors
    # Total: 5522 vectors

    candidates = []

    # Type 1: two ±1's
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)

    # Type 2: one ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)

    # Type 3: four ±1's
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
                candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)

    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)

    # Conflict condition: dist^2 < max_norm^2
    # For a set containing at least one norm^2=4 vector, max_norm^2 = 4
    # So we need dist^2 >= 4 for all pairs.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u·v
    # For two norm^2=4 vectors: dist^2 = 8 - 2 dot. Conflict if 8-2dot < 4 => dot > 2 => dot >= 3
    # For two norm^2=2 vectors: dist^2 = 4 - 2 dot. Conflict if 4-2dot < 4 => dot > 0 => dot >= 1
    # For norm^2=2 and norm^2=4: dist^2 = 6 - 2 dot. Conflict if 6-2dot < 4 => dot > 1 => dot >= 2

    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)

    best_set = None
    best_size = 0

    # Strategy 1: Greedy Independent Set with Random Restarts
    np.random.seed(42)
    max_restarts = 300

    for attempt in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Weighted greedy: pick node with minimum degree, with random tie-breaking
        # Use a priority queue approach for efficiency
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).astype(np.int32)

        selected = []

        while remaining.any():
            if time.time() > deadline - 10:
                break

            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break

            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]

            # Random choice among minimum degree nodes
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]

            selected.append(chosen_global)

            # Remove chosen and its neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining

            if np.any(actual_remove):
                # Update degrees for remaining nodes
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease

            remaining[actual_remove] = False

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()

    # Strategy 2: Local Search / Swap Improvement
    if best_set is not None and time.time() < deadline - 5:
        selected_set = set(best_set)
        # Create a fast lookup structure
        selected_arr = np.array(best_set)

        # Build conflict lookup for selected nodes
        # For each node, we want to know which selected nodes conflict with it
        # We can precompute this

        improved = True
        max_improvement_rounds = 50

        for round_num in range(max_improvement_rounds):
            if time.time() > deadline - 5:
                break

            improved = False

            # Try to find a node that can be added by removing one selected node
            # For efficiency, we look at nodes that conflict with exactly one selected node
            # and check if that selected node can be removed

            # Compute for each non-selected node, how many selected nodes it conflicts with
            non_selected = np.ones(n, dtype=bool)
            non_selected[best_set] = False

            if not non_selected.any():
                break

            # Get conflict counts with selected nodes for non-selected nodes
            # conflict is n x n, selected is list
            # For each non-selected node i, count = sum(conflict[i, j] for j in best_set)

            # Vectorized: conflict[non_selected_indices][:, selected_indices].sum(axis=1)
            non_sel_indices = np.where(non_selected)[0]
            sel_indices = np.array(best_set)

            # This can be memory intensive for large n, but n=5522 is manageable
            conflict_counts = conflict[non_sel_indices][:, sel_indices].sum(axis=1)

            # Find non-selected nodes that conflict with exactly 1 selected node
            single_conflict_mask = conflict_counts == 1
            if single_conflict_mask.any():
                candidate_indices = non_sel_indices[single_conflict_mask]

                for cand in candidate_indices:
                    if time.time() > deadline - 5:
                        break

                    # Find which selected node it conflicts with
                    conflicting_sel_idx = np.where(conflict[cand, sel_indices])[0]
                    if len(conflicting_sel_idx) != 1:
                        continue

                    remove_idx = conflicting_sel_idx[0]
                    remove_node = sel_indices[remove_idx]

                    # Try the swap: remove remove_node, add cand
                    # Check if this leads to improvement by seeing if we can add more nodes
                    # For now, just do the swap if it doesn't create new conflicts
                    # (it shouldn't, since cand only conflicts with remove_node among selected)

                    # Perform swap
                    best_set.remove(remove_node)
                    best_set.append(cand)
                    selected_set.discard(remove_node)
                    selected_set.add(cand)
                    improved = True

                    if len(best_set) > best_size:
                        best_size = len(best_set)

                    # After a swap, we need to recompute, so break inner loop
                    break

            if not improved:
                break

    # Strategy 3: Try to add more nodes greedily after local search
    if best_set is not None and time.time() < deadline - 3:
        selected_set = set(best_set)
        # Sort candidates by degree (prefer low degree)
        deg_order = np.argsort(conflict.sum(axis=1))

        for idx in deg_order:
            if time.time() > deadline - 3:
                break

            if idx in selected_set:
                continue

            # Check if idx conflicts with any selected node
            if not conflict[idx, list(selected_set)].any():
                best_set.append(idx)
                selected_set.add(idx)
                best_size = len(best_set)

    # Final verification and return
    if best_set is None:
        # Fallback: return all Type 1 vectors (they satisfy dist^2 >= 2 = max_norm^2 for Type 1 only)
        # But this might not be valid if mixed. Let's just return empty or a small valid set.
        # Actually, the greedy should have found something. If not, return first 220 (Type 1)
        # Type 1 vectors: dist^2 between any two is 0, 2, or 4.
        # If we pick all 220, min dist^2 is 2 (for e.g. e1+e2 and e1-e2, diff=2e2, norm^2=4? No.
        # e1+e2 = [1,1,0..], e1-e2 = [1,-1,0..]. diff = [0,2,0..], norm^2=4.
        # e1+e2 and e2+e3 = [0,1,1,0..]. diff = [1,0,-1,0..], norm^2=2.
        # So min dist^2 is 2. Max norm^2 is 2. So 2>=2 is valid.
        # So all 220 Type 1 vectors are valid.
        return candidates[:220]

    result = candidates[np.array(best_set)]

    # Verify the result
    if len(result) > 1:
        norms = np.linalg.norm(result, axis=1)
        max_norm = np.max(norms)
        # Check min pairwise distance
        # For efficiency, we only check if the set is large
        if len(result) < 2000:
            dists = np.linalg.norm(result[:, None, :] - result[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            if min_dist < max_norm - 1e-9:
                # Invalid, fall back to Type 1 only
                return candidates[:220]

    return result
