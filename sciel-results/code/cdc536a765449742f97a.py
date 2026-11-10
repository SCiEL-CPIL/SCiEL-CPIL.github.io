import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin for 330s limit

    num_points = N * N

    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(num_points)]

    # Precompute squared distances between all pairs
    # dist2[i][j] = squared distance from point i to point j
    # We'll store as a list of arrays for faster access
    dist2 = [None] * num_points
    for i in range(num_points):
        xi, yi = coords[i]
        row = [0] * num_points
        for j in range(num_points):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            row[j] = dx * dx + dy * dy
        dist2[i] = row

    def _validate(selected):
        """Check if the set is valid (no isosceles triangles)."""
        n = len(selected)
        if n <= 2:
            return True
        for i in range(n):
            pi = selected[i]
            seen = set()
            for j in range(n):
                if i == j:
                    continue
                d = dist2[pi][selected[j]]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def _can_add(selected_set, dist_used, p_idx):
        """
        Check if p_idx can be added to the set.
        dist_used[p] = set of distances already used at point p (from selected points to p)
        We need: for all q in selected, dist2[p][q] not in dist_used[p] AND dist2[q][p] not in dist_used[q]
        """
        if p_idx in selected_set:
            return False
        for q in selected_set:
            d = dist2[p_idx][q]
            if d in dist_used[p_idx] or d in dist_used[q]:
                return False
        return True

    def _greedy_with_priority(seed):
        """Greedy construction with priority based on remaining conflicts."""
        rng = random.Random(seed)
        selected_set = set()
        dist_used = [set() for _ in range(num_points)]
        selected = []

        # Start with random order
        candidates = list(range(num_points))
        rng.shuffle(candidates)

        for p_idx in candidates:
            if _can_add(selected_set, dist_used, p_idx):
                selected_set.add(p_idx)
                selected.append(p_idx)
                dist_used[p_idx] = set()
                for q in selected_set:
                    if q == p_idx:
                        continue
                    d = dist2[p_idx][q]
                    dist_used[p_idx].add(d)
                    dist_used[q].add(d)

        return selected, selected_set, dist_used

    def _local_search(selected, selected_set, dist_used, budget_sec=2.0):
        """
        Local search: try to add points. If stuck, remove a point that's causing
        the most conflicts, then try to re-add.
        """
        start = time.time()
        best_size = len(selected)
        best_selected = selected[:]
        best_set = selected_set.copy()
        best_dist_used = [s.copy() for s in dist_used]
        rng = random.Random()

        while time.time() - start < budget_sec:
            # Try to add random unselected points
            added = False
            for _ in range(300):
                if time.time() - start >= budget_sec:
                    break
                p_idx = rng.randint(0, num_points - 1)
                if p_idx not in selected_set:
                    # Check if we can add
                    can_add = True
                    for q in selected_set:
                        d = dist2[p_idx][q]
                        if d in dist_used[p_idx] or d in dist_used[q]:
                            can_add = False
                            break
                    if can_add:
                        # Add it
                        selected_set.add(p_idx)
                        selected.append(p_idx)
                        dist_used[p_idx] = set()
                        for q in selected_set:
                            if q == p_idx:
                                continue
                            d = dist2[p_idx][q]
                            dist_used[p_idx].add(d)
                            dist_used[q].add(d)
                        added = True
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_selected = selected[:]
                            best_set = selected_set.copy()
                            best_dist_used = [s.copy() for s in dist_used]

            if not added and selected:
                # Find a point to remove: one with the most "conflict potential"
                # Remove a random point
                idx = rng.randint(0, len(selected) - 1)
                p_idx = selected[idx]
                # Remove from data structures
                for q in selected_set:
                    if q != p_idx:
                        d = dist2[p_idx][q]
                        dist_used[q].discard(d)
                selected_set.discard(p_idx)
                selected.remove(p_idx)
                dist_used[p_idx] = set()

        # Restore best
        if len(selected) != best_size:
            selected = best_selected
            selected_set = best_set
            dist_used = best_dist_used

        return best_size, best_selected, best_set, best_dist_used

    def _conflict_aware_greedy(seed):
        """
        Greedy that prioritizes points that have fewer conflicts with already-selected points.
        """
        rng = random.Random(seed)
        selected_set = set()
        dist_used = [set() for _ in range(num_points)]
        selected = []

        candidates = list(range(num_points))
        rng.shuffle(candidates)

        # Process in order, but skip points that conflict
        for p_idx in candidates:
            if _can_add(selected_set, dist_used, p_idx):
                selected_set.add(p_idx)
                selected.append(p_idx)
                for q in selected_set:
                    if q == p_idx:
                        continue
                    d = dist2[p_idx][q]
                    dist_used[p_idx].add(d)
                    dist_used[q].add(d)

        return selected, selected_set, dist_used

    def _pairwise_repair(selected, selected_set, dist_used, budget_sec=2.0):
        """
        Repair-based search: find conflicting triples and remove the least "valuable" point.
        """
        start = time.time()
        best_size = len(selected)
        best_selected = selected[:]
        best_set = selected_set.copy()
        best_dist_used = [s.copy() for s in dist_used]
        rng = random.Random()

        while time.time() - start < budget_sec:
            # Try to add points
            added = False
            for _ in range(200):
                if time.time() - start >= budget_sec:
                    break
                p_idx = rng.randint(0, num_points - 1)
                if p_idx not in selected_set:
                    can_add = True
                    for q in selected_set:
                        d = dist2[p_idx][q]
                        if d in dist_used[p_idx] or d in dist_used[q]:
                            can_add = False
                            break
                    if can_add:
                        selected_set.add(p_idx)
                        selected.append(p_idx)
                        for q in selected_set:
                            if q == p_idx:
                                continue
                            d = dist2[p_idx][q]
                            dist_used[p_idx].add(d)
                            dist_used[q].add(d)
                        added = True
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_selected = selected[:]
                            best_set = selected_set.copy()
                            best_dist_used = [s.copy() for s in dist_used]

            if not added and selected:
                # Find a point whose removal allows the most additions
                # Heuristic: remove a point with high degree in the conflict graph
                # For now, remove random
                idx = rng.randint(0, len(selected) - 1)
                p_idx = selected[idx]
                for q in selected_set:
                    if q != p_idx:
                        d = dist2[p_idx][q]
                        dist_used[q].discard(d)
                selected_set.discard(p_idx)
                selected.remove(p_idx)
                dist_used[p_idx] = set()

        if len(selected) != best_size:
            selected = best_selected
            selected_set = best_set
            dist_used = best_dist_used

        return best_size, best_selected, best_set, best_dist_used

    # Main search
    best_size = 0
    best_selected = []
    best_set = set()
    best_dist_used = None

    # Strategy 1: Multiple greedy constructions
    for seed in range(30):
        if time.time() > deadline:
            break
        selected, sel_set, dist_u = _greedy_with_priority(seed)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
            best_set = sel_set.copy()
            best_dist_used = [s.copy() for s in dist_u]

    # Strategy 2: Local search from best greedy
    if best_selected and time.time() < deadline - 30:
        budget = min(20.0, max(5.0, deadline - time.time() - 10))
        best_size, best_selected, best_set, best_dist_used = _local_search(
            best_selected, best_set, best_dist_used, budget_sec=budget
        )

    # Strategy 3: More greedy + local search with different seeds
    for seed in range(10, 25):
        if time.time() > deadline - 10:
            break
        selected, sel_set, dist_u = _conflict_aware_greedy(seed)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
            best_set = sel_set.copy()
            best_dist_used = [s.copy() for s in dist_u]
        # Quick local search
        if time.time() < deadline - 5:
            best_size2, best_selected2, best_set2, best_dist_used2 = _local_search(
                selected, sel_set, dist_u, budget_sec=1.5
            )
            if best_size2 > best_size:
                best_size = best_size2
                best_selected = best_selected2
                best_set = best_set2
                best_dist_used = best_dist_used2

    # Strategy 4: Final local search on best
    if best_selected and time.time() < deadline - 5:
        budget = min(15.0, max(3.0, deadline - time.time() - 3))
        best_size, best_selected, best_set, best_dist_used = _local_search(
            best_selected, best_set, best_dist_used, budget_sec=budget
        )

    # Validate
    if not _validate(best_selected):
        # Fallback
        best_selected = list(range(min(10, num_points)))

    return [(i // N, i % N) for i in best_selected]

# EVOLVE-BLOCK-END
