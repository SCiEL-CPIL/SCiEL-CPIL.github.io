"""Improved program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # --- W4 vectors (weight 4, entries in {-1,0,1}, norm^2 = 4) ---
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # --- W1 vectors (weight 1, entries in {-2,0,2}, norm^2 = 4) ---
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # --- Build support graph ---
    # Two W4 vectors conflict if their inner product is >= 2 (distance < 2*sqrt(4))
    # For W4 vectors with entries in {-1,0,1}, inner product = sum of products over common supports
    # If they share 4 positions with same sign: inner product = 4 (conflict)
    # If they share 3 positions with same sign: inner product = 3 (conflict, dist^2 = 8-6=2 < 8)
    # Actually let's compute: dist^2 = |v|^2 + |w|^2 - 2*v.w = 4 + 4 - 2*v.w = 8 - 2*v.w
    # We need dist >= max_norm = 2*sqrt(4) = 4? No, max norm is sqrt(4) = 2 for W4, but W1 has norm 2 too.
    # Wait, all vectors have norm^2 = 4, so norm = 2.
    # We need min pairwise distance >= max norm = 2.
    # dist^2 = 8 - 2*v.w >= 4, so v.w <= 2.
    # For W4-W4: v.w is sum over intersection. Max |v.w| when supports overlap.
    # If supports are identical (4 common positions), v.w can be -4,-2,0,2,4.
    # v.w <= 2 means we need v.w <= 2. So v.w = 4 conflicts, v.w = -4 is fine (dist^2=16).
    # Actually v.w can be negative too. Let's be careful.
    # dist^2 = 8 - 2*v.w. We need dist^2 >= 4, i.e., 8 - 2*v.w >= 4, i.e., v.w <= 2.
    # So conflict when v.w > 2, i.e., v.w >= 3.
    # For W4 vectors with entries in {-1,0,1}, v.w is integer in {-4,-3,...,3,4}.
    # v.w >= 3 means conflict.

    # Group by support
    support_to_indices = defaultdict(list)
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        support_to_indices[support].append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)

    # For the same support, two vectors conflict if their inner product >= 3
    # With 4 positions and signs in {-1,1}, inner product = sum of 4 products, each in {-1,1}
    # So inner product is in {-4,-2,0,2,4} (even values only since 4 terms)
    # Actually: each product is +1 or -1, sum of 4 such = -4,-2,0,2,4
    # v.w >= 3 means v.w = 4, which happens when all 4 signs match.
    # So for same support, only the identical vector conflicts (v.w=4).
    # Wait, that means for same support, all pairs except identical have v.w <= 2.
    # Let's verify: signs (1,1,1,1) and (1,1,1,-1): dot = 1+1+1-1 = 2. OK, no conflict.
    # signs (1,1,1,1) and (1,1,-1,-1): dot = 1+1-1-1 = 0. OK.
    # So for same support, only the exact same vector conflicts with itself.
    # That means we can include at most one vector per support? No, we can include multiple
    # as long as they're not identical. But we're selecting from the full set.

    # For different supports, we need to check inner product.
    # Two supports with intersection size k: inner product is sum over k positions.
    # If k=4 (same support): handled above.
    # If k=3: inner product in {-3,-1,1,3}. Conflict if >= 3, i.e., = 3.
    # If k=2: inner product in {-2,0,2}. No conflict possible.
    # If k=1: inner product in {-1,1}. No conflict.
    # If k=0: inner product = 0. No conflict.

    # So conflicts only occur when:
    # 1. Same support AND identical vector (v.w = 4)
    # 2. Different supports with intersection size 3 AND all 3 common signs match (v.w = 3)

    # Build conflict graph on individual vectors
    # For efficiency, work with support-level graph first
    # Support i and support j (different) conflict if |intersection| >= 3
    # For same support, all vectors are mutually non-conflicting except identical ones

    # Let's build the full vector-level conflict graph
    # N4 = C(11,4) * 16 = 330 * 16 = 5280 vectors
    # That's a 5280x5280 matrix, which is ~22M entries. Manageable.

    # But let's be smarter. For each vector, its conflicts are:
    # - Itself (but we don't count self)
    # - Other vectors with same support: none (as shown above)
    # - Vectors with different support sharing >= 3 positions with matching signs

    # Let's precompute for each support, which other supports share >= 3 positions
    support_sets = [set(s) for s in supports]
    support_neighbors = [set() for _ in range(n_supports)]
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            if len(support_sets[i] & support_sets[j]) >= 3:
                support_neighbors[i].add(j)
                support_neighbors[j].add(i)

    # For each vector, compute its conflict set
    # Vector i with support s_i and signs sig_i conflicts with vector j if:
    # - s_j shares >= 3 positions with s_i AND the signs match on all shared positions
    # (since if they share exactly 3, we need all 3 to match for v.w=3)

    # Actually wait: if they share 3 positions, v.w is the sum over those 3 positions.
    # Each product is +1 or -1. Sum is in {-3,-1,1,3}.
    # v.w = 3 requires all 3 products to be +1, i.e., all signs match on the 3 shared positions.
    # v.w = -3 is fine (dist^2 = 8-2(-3) = 14 >= 4).

    # So conflict condition for different supports: share exactly 3 positions AND signs match on all 3.
    # (If they share 4, it's the same support case.)

    # Let's build conflict lists for each vector
    # For speed, use the support structure
    vector_support = np.array([supports.index(tuple(np.nonzero(vectors_v4[i])[0])) for i in range(N4)])
    # This is slow. Let's build it differently.

    # Map from support index to list of vector indices
    sup_to_vecs = {si: support_to_indices[supports[si]] for si in range(n_supports)}

    # For each vector, determine its sign pattern on its support
    # And for each neighboring support, check sign compatibility

    # Precompute: for each support pair (i,j) with |intersection|>=3,
    # which sign combinations conflict?
    # This is complex. Let's just build the full conflict matrix using vectorized ops.

    # Alternative: since N4 = 5280, let's compute all pairwise dot products
    # using matrix multiplication, then threshold.
    # v4 @ v4.T gives all dot products. 5280x5280 = ~28M floats. Fine.

    dot_products = vectors_v4 @ vectors_v4.T
    # Conflict when dot > 2 (i.e., dot >= 3)
    conflict_matrix = dot_products > 2
    np.fill_diagonal(conflict_matrix, False)

    # Add W1 vectors: they have norm^2=4, dot with W4 is at most 2 (one position overlap)
    # W1-W1: two different W1 vectors have dot=0 (different positions) or dot=4 (same position, same sign)
    # or dot=-4 (same position, opposite sign).
    # For same position same sign: dot=4, dist^2=8-8=0. Conflict!
    # For same position opposite sign: dot=-4, dist^2=8+8=16. Fine.
    # For different positions: dot=0, dist^2=8. Fine.
    # So W1 vectors: can't have two with same position and same sign.
    # We have 22 W1 vectors (11 positions * 2 signs).
    # Max independent set of W1: one per position, so 11.
    # Choose all 11 with sign +1 (or any consistent choice).

    # W4-W1: dot product is at most 1 (if the W1's position is in W4's support and signs match)
    # or -1. So |dot| <= 1. dist^2 = 8 - 2*dot >= 8-2=6 >= 4. Always fine.
    # So W1 vectors never conflict with W4 vectors.

    # W1-W1: conflict only if same position AND same sign.
    # So we can pick at most one W1 per position. Pick all 11 with +1 sign.
    # That gives 11 W1 vectors, all mutually non-conflicting.

    # So total = (max independent set of W4 conflict graph) + 11

    # Now solve maximum independent set on the W4 conflict graph
    # N4 = 5280 vertices. This is a large graph.
    # Use greedy + local search.

    # Convert to adjacency list for efficiency
    neighbors = [np.where(conflict_matrix[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(12345)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees: for each remaining node, subtract number of neighbors removed
                # This is expensive. Use a simpler approach.
                pass  # Skip degree update for speed; greedy still works

        return in_set

    def greedy_with_degree_update():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.astype(np.int64).copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = int(np.min(degrees[alive_indices]))
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each remaining node, count how many of its neighbors were in to_remove
                # Use conflict_matrix for this
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)

        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        cooling = 0.999
        min_T = 0.01

        # Count conflicts for each node
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            move_type = rng.random()

            if move_type < 0.60:
                # Add move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                weights = 1.0 / (1.0 + count_conflicts[not_in_set].astype(np.float64))
                weights /= weights.sum()
                s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                k = count_conflicts[s]
                delta = 1 - k
                if delta > 0 or rng.random() < np.exp(delta / T):
                    to_remove = np.where(conflict_matrix[s] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors[s]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85:
                # Swap move
                in_set_nodes = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_nodes) == 0 or len(not_in_set) == 0:
                    continue
                s = not_in_set[rng.integers(len(not_in_set))]
                k = count_conflicts[s]
                if k == 0:
                    continue
                nbrs_in_set = np.where(conflict_matrix[s] & in_set)[0]
                if len(nbrs_in_set) == 0:
                    continue
                u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                # Remove u, add s (and remove s's other conflicts)
                in_set[u] = False
                count_conflicts[neighbors[u]] -= 1
                to_remove = np.where(conflict_matrix[s] & in_set)[0]
                for v in to_remove:
                    in_set[v] = False
                    count_conflicts[neighbors[v]] -= 1
                in_set[s] = True
                count_conflicts[neighbors[s]] += 1
                current_size = int(in_set.sum())
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1

            else:
                # 2-for-2 move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) < 2:
                    continue
                i1 = rng.integers(len(not_in_set))
                i2 = rng.integers(len(not_in_set))
                if i1 == i2:
                    continue
                s1 = not_in_set[i1]
                s2 = not_in_set[i2]
                if conflict_matrix[s1, s2]:
                    continue
                c1 = np.where(conflict_matrix[s1] & in_set)[0]
                c2 = np.where(conflict_matrix[s2] & in_set)[0]
                union = np.union1d(c1, c2)
                delta = 2 - len(union)
                if delta > 0 or rng.random() < np.exp(delta / T):
                    for u in union:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    in_set[s1] = True
                    in_set[s2] = True
                    count_conflicts[neighbors[s1]] += 1
                    count_conflicts[neighbors[s2]] += 1
                    current_size = int(in_set.sum())
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1

            if stuck_counter > 3000:
                T = max(T, 5.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def polish(in_set_init, max_time=10.0):
        """Deterministic local improvement."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        polish_deadline = time.time() + max_time
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Pass A: add free nodes
            free = np.where((~in_set) & (count_conflicts == 0))[0]
            for s in free:
                in_set[s] = True
                current_size += 1
                count_conflicts[neighbors[s]] += 1
            if len(free) > 0:
                improved = True

            # Pass B: best single swap
            best_delta = 0
            best_s = -1
            for s in np.where(~in_set)[0]:
                k = count_conflicts[s]
                if k == 0:
                    continue
                d = 1 - k
                if d > best_delta:
                    best_delta = d
                    best_s = s
            if best_s >= 0 and best_delta > 0:
                to_remove = np.where(conflict_matrix[best_s] & in_set)[0]
                for u in to_remove:
                    in_set[u] = False
                    count_conflicts[neighbors[u]] -= 1
                in_set[best_s] = True
                current_size += best_delta
                count_conflicts[neighbors[best_s]] += 1
                improved = True

            # Pass C: best 2-for-2 swap (limited)
            not_in = np.where(~in_set)[0]
            if len(not_in) >= 2:
                sorted_not_in = not_in[np.argsort(count_conflicts[not_in])]
                max_pairs = min(len(sorted_not_in), 50)
                best_pair_delta = 0
                best_pair = None
                for idx1 in range(max_pairs):
                    s1 = sorted_not_in[idx1]
                    for idx2 in range(idx1 + 1, max_pairs):
                        s2 = sorted_not_in[idx2]
                        if conflict_matrix[s1, s2]:
                            continue
                        c1 = np.where(conflict_matrix[s1] & in_set)[0]
                        c2 = np.where(conflict_matrix[s2] & in_set)[0]
                        union = np.union1d(c1, c2)
                        delta = 2 - len(union)
                        if delta > best_pair_delta:
                            best_pair_delta = delta
                            best_pair = (s1, s2)
                if best_pair is not None and best_pair_delta > 0:
                    s1, s2 = best_pair
                    c1 = np.where(conflict_matrix[s1] & in_set)[0]
                    c2 = np.where(conflict_matrix[s2] & in_set)[0]
                    union = np.union1d(c1, c2)
                    for u in union:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    in_set[s1] = True
                    in_set[s2] = True
                    current_size += best_pair_delta
                    count_conflicts[neighbors[s1]] += 1
                    count_conflicts[neighbors[s2]] += 1
                    improved = True

        return in_set, current_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 300

    # First: greedy
    if time.time() < deadline - 10:
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining = deadline - 10 - time.time()
        if remaining > 5:
            sa_time = min(remaining * 0.5, 15.0)
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

            remaining = deadline - 5 - time.time()
            if remaining > 3:
                polish_time = min(remaining * 0.7, 12.0)
                p_result, p_size = polish(best_in_set, max_time=polish_time)
                if p_size > best_size:
                    best_size = p_size
                    best_in_set = p_result.copy()

    # More restarts
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break

        sa_time = min(remaining * 0.4, 10.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        remaining = deadline - 5 - time.time()
        if remaining > 4:
            polish_time = min(remaining * 0.5, 10.0)
            if polish_time > 2:
                p_result, p_size = polish(best_in_set, max_time=polish_time)
                if p_size > best_size:
                    best_size = p_size
                    best_in_set = p_result.copy()

    # Final polish
    final_time = max(0.0, deadline - 2 - time.time())
    if final_time > 1:
        fp_result, fp_size = polish(best_in_set, max_time=final_time)
        if fp_size > best_size:
            best_size = fp_size
            best_in_set = fp_result.copy()

    # Second final polish
    final_time2 = max(0.0, deadline - 1 - time.time())
    if final_time2 > 1:
        fp_result2, fp_size2 = polish(best_in_set, max_time=final_time2)
        if fp_size2 > best_size:
            best_size = fp_size2
            best_in_set = fp_result2.copy()

    # Assemble result: selected W4 vectors + 11 W1 vectors (one per position, sign +1)
    selected_v4 = vectors_v4[best_in_set]
    w1_selected = np.array([[2 if i == pos else 0 for i in range(D)] for pos in range(D)], dtype=np.int64)

    final_vectors = np.vstack([selected_v4, w1_selected])

    return final_vectors

# EVOLVE-BLOCK-END
