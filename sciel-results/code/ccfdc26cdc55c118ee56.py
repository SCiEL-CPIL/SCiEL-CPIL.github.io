import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590.0  # safety margin before 630s kill

    # Generate candidate vectors
    def gen_weight2():
        pts = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        pts.append(v)
        return np.array(pts, dtype=np.int64)

    def gen_weight3():
        pts = []
        for combo in combinations(range(d), 3):
            for mask in range(8):
                v = np.zeros(d, dtype=np.int64)
                for idx, pos in enumerate(combo):
                    v[pos] = 1 if (mask & (1 << idx)) else -1
                pts.append(v)
        return np.array(pts, dtype=np.int64)

    # Build conflict graph: edge (i,j) if d(i,j)^2 < max(||i||^2, ||j||^2)
    def build_conflict(points):
        n = len(points)
        norms = np.einsum('ij,ij->i', points, points).astype(np.int64)
        # Squared distance matrix
        dist_sq = norms[:, None] + norms[None, :] - 2 * (points @ points.T)
        # Conflict: dist_sq[i,j] < max(norms[i], norms[j])
        max_norms = np.maximum(norms[:, None], norms[None, :])
        conflict = (dist_sq < max_norms) | (dist_sq == 0)  # also exclude self/duplicates
        np.fill_diagonal(conflict, False)
        # Build adjacency bitsets (use Python ints for arbitrary precision)
        adj = [0] * n
        for i in range(n):
            bits = 0
            row = conflict[i]
            for j in range(n):
                if row[j] and j != i:
                    bits |= (1 << j)
            adj[i] = bits
        return adj, norms

    # Maximal Independent Set via greedy (for upper bound and initial solution)
    def greedy_mis(adj, available, order=None):
        """Greedy MIS: repeatedly pick vertex with fewest available neighbors."""
        n = len(adj)
        if available == 0:
            return 0, 0
        # Count available neighbors for each vertex
        best_v = -1
        best_deg = -1
        # Try to find vertex with minimum degree in available subgraph
        avail_list = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            avail_list.append(v)
            temp ^= lsb

        if not avail_list:
            return 0, 0

        # Sort by degree in available subgraph
        degs = []
        for v in avail_list:
            deg = (adj[v] & available).bit_count()
            degs.append((deg, v))
        degs.sort()

        selected = []
        remaining = available
        for _, v in degs:
            if not (remaining & (1 << v)):
                continue
            selected.append(v)
            remaining &= ~(adj[v] | (1 << v))
            if remaining == 0:
                break
        return len(selected), selected

    # Branch and Bound MIS
    def bnb_mis(adj, available, best_count, best_set):
        """Branch and bound for maximum independent set."""
        if available == 0:
            return best_count, best_set

        # Upper bound: greedy packing
        ub_count, _ = greedy_mis(adj, available)
        if ub_count <= best_count:
            return best_count, best_set

        # Time check
        if time.time() > deadline - 5:
            return best_count, best_set

        # Pick a vertex to branch on (highest degree in available)
        temp = available
        best_v = -1
        best_deg = -1
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb

        if best_v < 0:
            return best_count, best_set

        # Branch 1: include best_v
        new_avail = available & ~(adj[best_v] | (1 << best_v))
        # Quick bound for this branch
        ub1, _ = greedy_mis(adj, new_avail)
        if ub1 + 1 > best_count:
            best_count, best_set = bnb_mis(adj, new_avail, best_count - 1, best_set)
            # Reconstruct: we need to track the set
            # Actually let's redo this with proper set tracking

        # Branch 2: exclude best_v
        new_avail2 = available & ~(1 << best_v)
        ub2, _ = greedy_mis(adj, new_avail2)
        if ub2 > best_count:
            best_count, best_set = bnb_mis(adj, new_avail2, best_count, best_set)

        return best_count, best_set

    # Better BnB with proper set tracking
    def bnb_mis_v2(adj, available, depth=0):
        """Returns (count, set_of_vertices) for maximum independent set in available."""
        if time.time() > deadline - 5:
            return 0, set()
        if available == 0:
            return 0, set()

        # Upper bound via greedy
        ub_count, ub_set = greedy_mis(adj, available)
        if ub_count == 0:
            return 0, set()

        # If upper bound is small, just return greedy
        if ub_count <= 2:
            return ub_count, set(ub_set)

        # Pick branching vertex (max degree in available)
        temp = available
        best_v = -1
        best_deg = -1
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb

        if best_v < 0:
            return 0, set()

        # Branch: include best_v
        avail_with = available & ~(adj[best_v] | (1 << best_v))
        c1, s1 = bnb_mis_v2(adj, avail_with)
        s1.add(best_v)
        count1 = len(s1)

        # Branch: exclude best_v
        avail_without = available & ~(1 << best_v)
        c2, s2 = bnb_mis_v2(adj, avail_without)
        count2 = len(s2)

        if count1 >= count2:
            return count1, s1
        else:
            return count2, s2

    # Greedy with multiple orderings (faster, good initial solution)
    def greedy_multi_order(points, norms, conflict, n, num_orders=20):
        """Try greedy MIS with different orderings, return best set."""
        best_set = set()
        best_count = 0

        for order_idx in range(num_orders):
            if time.time() > deadline - 10:
                break
            # Different orderings
            if order_idx == 0:
                order = list(range(n))
            elif order_idx == 1:
                order = sorted(range(n), key=lambda i: norms[i])
            elif order_idx == 2:
                order = sorted(range(n), key=lambda i: -norms[i])
            else:
                import random
                rng = random.Random(order_idx * 12345 + 67890)
                order = list(range(n))
                rng.shuffle(order)

            selected = set()
            # Greedy: add vertex if no conflict with already selected
            for v in order:
                # Check if v conflicts with any selected vertex
                compatible = True
                for u in selected:
                    if conflict[v, u]:
                        compatible = False
                        break
                if compatible:
                    selected.add(v)

            if len(selected) > best_count:
                best_count = len(selected)
                best_set = set(selected)

        return best_set, best_count

    # Local search: try to swap vertices to improve
    def local_search(points, norms, conflict, n, current_set, max_iters=500):
        """Try to improve by removing one and adding another."""
        if time.time() > deadline - 10:
            return current_set
        current_set = set(current_set)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 10:
                break
            improved = False
            iters += 1
            # Try removing each vertex and see if we can add a better one
            for rem in list(current_set):
                if time.time() > deadline - 10:
                    break
                candidate_set = current_set - {rem}
                # Try to add a vertex not in candidate_set that doesn't conflict
                for add in range(n):
                    if add in candidate_set:
                        continue
                    # Check conflicts
                    ok = True
                    for u in candidate_set:
                        if conflict[add, u]:
                            ok = False
                            break
                    if ok:
                        # Check if this improves (it won't in count, but might help future)
                        # Actually same count, so skip unless we want diversity
                        pass

        return current_set

    # Main: try different candidate sets
    best_points = None
    best_k = 0

    # Weight-2 candidates (220 points)
    pts_w2 = gen_weight2()
    n2 = len(pts_w2)
    norms2 = np.einsum('ij,ij->i', pts_w2, pts_w2).astype(np.int64)
    dist2 = norms2[:, None] + norms2[None, :] - 2 * (pts_w2 @ pts_w2.T)
    maxn2 = np.maximum(norms2[:, None], norms2[None, :])
    conflict2 = (dist2 < maxn2) | (dist2 == 0)
    np.fill_diagonal(conflict2, False)

    # For weight-2, all norms are 2, all distances between distinct points are 2 or 4
    # dist=2 means they share a coordinate with opposite sign or same coordinate
    # Actually for weight-2 vectors with norm 2:
    # Two vectors conflict if dist^2 < 2, i.e., dist^2 = 0 (same vector)
    # So all 220 weight-2 vectors are mutually compatible!
    # Wait: dist^2 between two weight-2 vectors:
    # If they share no coordinates: dist^2 = 4 (each has two ±1, no overlap)
    # If they share one coordinate: dist^2 = 2 (overlap contributes 0 or 2, non-overlap contributes 1+1=2)
    #   - Same sign at overlap: dist^2 = 0+1+1 = 2
    #   - Opposite sign at overlap: dist^2 = 4+1+1 = 6... wait
    # Let me recompute. v1 has 1 at i, 1 at j. v2 has 1 at i, 1 at k (j≠k).
    # dist^2 = (1-1)^2 + (1-1)^2 + (0-1)^2 + (0-1)^2 = 0+0+1+1 = 2
    # v1 = (1,1,0,...), v2 = (-1,1,0,...): dist^2 = (1-(-1))^2 + (1-1)^2 = 4+0 = 4
    # v1 = (1,1,0,...), v3 = (1,0,1,...): dist^2 = 0 + (1-0)^2 + (0-1)^2 = 2
    # So minimum distance is 2, and max norm is 2. Condition: min_dist >= max_norm => 2 >= 2. OK!
    # So all 220 weight-2 vectors form a valid configuration.

    # But can we do better? Add weight-3 vectors?
    # Weight-3 vectors have norm 3. For a weight-3 vector to be compatible with weight-2:
    # dist^2 >= max(2, 3) = 3, so dist^2 >= 4 (since integer)
    # dist^2 between weight-2 (norm 2) and weight-3 (norm 3):
    # = 2 + 3 - 2*(dot product) = 5 - 2*dot
    # dot product ranges from -3 to 3 (but limited by support overlap)
    # If supports disjoint: dot=0, dist^2=5 >= 4. OK.
    # If 1 coord overlap: dot in {-1, 0, 1}, dist^2 in {3, 5, 7}. dist^2=3 < 4. CONFLICT if dot=1.
    # If 2 coords overlap: dot in {-2,-1,0,1,2}, dist^2 in {1,3,5,7,9}. dist^2=1 or 3 < 4. CONFLICT.
    # So weight-3 is compatible with weight-2 only if dot product <= 0 (i.e., not both positive at any shared coord with matching signs)
    # More precisely: dot <= 0 means dist^2 >= 5.

    # This is getting complex. Let's use the general algorithm.

    # Try weight-2 + weight-3 combined
    pts_w3 = gen_weight3()
    pts_combined = np.vstack([pts_w2, pts_w3])
    n_comb = len(pts_combined)
    norms_comb = np.einsum('ij,ij->i', pts_combined, pts_combined).astype(np.int64)
    dist_comb = norms_comb[:, None] + norms_comb[None, :] - 2 * (pts_combined @ pts_combined.T)
    maxn_comb = np.maximum(norms_comb[:, None], norms_comb[None, :])
    conflict_comb = (dist_comb < maxn_comb) | (dist_comb == 0)
    np.fill_diagonal(conflict_comb, False)

    print(f"Combined candidates: {n_comb}, conflicts: {conflict_comb.sum()}")

    # Greedy on combined set
    t0 = time.time()
    best_set_comb, best_k_comb = greedy_multi_order(pts_combined, norms_comb, conflict_comb, n_comb, num_orders=50)
    print(f"Greedy combined: k={best_k_comb}, time={time.time()-t0:.1f}s")

    if best_k_comb > best_k:
        best_k = best_k_comb
        best_points = pts_combined[list(best_set_comb)]

    # Also try just weight-2 (should give 220)
    # All weight-2 are compatible, so k=220
    if n2 > best_k:
        best_k = n2
        best_points = pts_w2.copy()

    # Try weight-3 only
    n3 = len(pts_w3)
    norms3 = np.einsum('ij,ij->i', pts_w3, pts_w3).astype(np.int64)
    dist3 = norms3[:, None] + norms3[None, :] - 2 * (pts_w3 @ pts_w3.T)
    maxn3 = np.maximum(norms3[:, None], norms3[None, :])
    conflict3 = (dist3 < maxn3) | (dist3 == 0)
    np.fill_diagonal(conflict3, False)
    print(f"Weight-3 candidates: {n3}, conflicts: {conflict3.sum()}")

    t0 = time.time()
    best_set_w3, best_k_w3 = greedy_multi_order(pts_w3, norms3, conflict3, n3, num_orders=50)
    print(f"Greedy weight-3: k={best_k_w3}, time={time.time()-t0:.1f}s")

    if best_k_w3 > best_k:
        best_k = best_k_w3
        best_points = pts_w3[list(best_set_w3)]

    # Try weight-2 + weight-3 with more greedy orders and local refinement
    if time.time() < deadline - 30:
        t0 = time.time()
        best_set_comb2, best_k_comb2 = greedy_multi_order(pts_combined, norms_comb, conflict_comb, n_comb, num_orders=200)
        print(f"Greedy combined (200 orders): k={best_k_comb2}, time={time.time()-t0:.1f}s")
        if best_k_comb2 > best_k:
            best_k = best_k_comb2
            best_points = pts_combined[list(best_set_comb2)]

    # Try to improve with a simple local search on the combined set
    if time.time() < deadline - 20 and best_k > 0:
        # Convert best_set to indices in combined set
        # Try adding more points greedily
        if best_points is not None:
            # Check if we can add any remaining points
            current_indices = set()
            if len(best_points) > 0:
                # Find which combined indices are in best_points
                for i in range(n_comb):
                    if np.array_equal(pts_combined[i], best_points[0]) and len(best_points) == 1:
                        current_indices.add(i)
                    elif len(best_points) > 1:
                        for j in range(len(best_points)):
                            if np.array_equal(pts_combined[i], best_points[j]):
                                current_indices.add(i)
                                break

            # Simpler: just do greedy from scratch with more orders
            if time.time() < deadline - 15:
                best_set_comb3, best_k_comb3 = greedy_multi_order(pts_combined, norms_comb, conflict_comb, n_comb, num_orders=500)
                if best_k_comb3 > best_k:
                    best_k = best_k_comb3
                    best_points = pts_combined[list(best_set_comb3)]

    # Fallback
    if best_points is None or best_k < 2:
        best_points = pts_w2
        best_k = n2

    return np.array(best_points, dtype=np.int64)
