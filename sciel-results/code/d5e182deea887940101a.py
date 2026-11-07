import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Precompute adjacency as bitmasks for fast intersection checks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))

    def make_mask(vertex_list):
        m = 0
        for v in vertex_list:
            m |= (1 << v)
        return m

    def mask_to_list(m):
        return [v for v in range(n) if (m >> v) & 1]

    def edges_of_biclique(L, R):
        """Return set of normalized edges covered by biclique (L, R)."""
        es = set()
        for u in L:
            for v in R:
                es.add((min(u, v), max(u, v)))
        return es

    def is_valid_biclique(L, R):
        """Check every cross pair is an edge."""
        for u in L:
            for v in R:
                if not (adj_mask[u] >> v) & 1:
                    return False
        return True

    def biclique_edges_set(L, R):
        es = set()
        for u in L:
            for v in R:
                es.add((min(u, v), max(u, v)))
        return es

    # --- Initial solution: greedy maximal biclique extraction ---
    def greedy_init():
        rem = set(edge_set)
        bics = []
        while rem:
            # Pick edge with fewest possible extensions (most constrained)
            best_edge = None
            best_score = float('inf')
            for e in rem:
                u, v = e
                # Count how many vertices could join L or R
                score = 0
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if (adj_mask[x] >> v) & 1 and (min(x, v), max(x, v)) in rem:
                        score += 1
                    if (adj_mask[x] >> u) & 1 and (min(x, u), max(x, u)) in rem:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_edge = e
            u, v = best_edge
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L:
                        continue
                    if all((adj_mask[x] >> y) & 1 for y in R) and all((min(x, y), max(x, y)) in rem for y in R):
                        L.add(x)
                        changed = True
                for y in range(n):
                    if y in R:
                        continue
                    if all((adj_mask[x] >> y) & 1 for x in L) and all((min(x, y), max(x, y)) in rem for x in L):
                        R.add(y)
                        changed = True
            b_edges = set((min(a, b), max(a, b)) for a in L for b in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

    def validate_and_count(bics):
        """Check validity; return (is_valid, count, covered_edges)."""
        covered = set()
        for L, R in bics:
            if not L or not R:
                return False, 0, covered
            for u in L:
                for v in R:
                    if not (adj_mask[u] >> v) & 1:
                        return False, 0, covered
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return False, 0, covered
                    covered.add(e)
        if covered != edge_set:
            return False, 0, covered
        return True, len(bics), covered

    # --- Simulated Annealing ---
    def can_merge_masks(L1m, R1m, L2m, R2m):
        """Check if merging two bicliques (as masks) is valid."""
        # L1 x R2 must all be edges
        for u in mask_to_list(L1m):
            if (adj_mask[u] & R2m) != R2m:
                return False
        # L2 x R1 must all be edges
        for u in mask_to_list(L2m):
            if (adj_mask[u] & R1m) != R1m:
                return False
        return True

    def try_merge(bics, i, j):
        """Try to merge biclique i and j. Return new list or None."""
        L1, R1 = bics[i]
        L2, R2 = bics[j]
        L1m, R1m = make_mask(L1), make_mask(R1)
        L2m, R2m = make_mask(L2), make_mask(R2)
        if not can_merge_masks(L1m, R1m, L2m, R2m):
            return None
        new_L = mask_to_list(L1m | L2m)
        new_R = mask_to_list(R1m | R2m)
        if not new_L or not new_R:
            return None
        new_bics = [b for k, b in enumerate(bics) if k != i and k != j]
        new_bics.append([new_L, new_R])
        return new_bics

    def try_split(bics, i):
        """Try to split biclique i into two by moving a vertex from L to R or vice versa.
        This is a 2-opt style move: pick a vertex from L, move to R (or vice versa),
        check if both resulting bicliques are valid and together still cover same edges."""
        L, R = bics[i]
        if len(L) < 2 or len(R) < 2:
            return None
        # Try moving one vertex from L to R
        for idx, v in enumerate(L):
            new_L = L[:idx] + L[idx+1:]
            new_R = R + [v]
            if not new_L or not new_R:
                continue
            if is_valid_biclique(new_L, new_R):
                # The other part: we need to keep the same edge coverage.
                # Actually splitting means: original edges = edges(new_L,new_R) ∪ edges(other_L, other_R)
                # This is complex; simpler: just try swapping a vertex between L and R
                pass
        # Try swapping: move v from L to R and w from R to L
        for idx, v in enumerate(L):
            for jdx, w in enumerate(R):
                new_L = L[:idx] + L[idx+1:] + [w]
                new_R = R[:jdx] + R[jdx+1:] + [v]
                if not new_L or not new_R:
                    continue
                # Check both are valid bicliques
                if is_valid_biclique(new_L, new_R):
                    # Check edge coverage is preserved
                    orig_edges = biclique_edges_set(L, R)
                    new_edges = biclique_edges_set(new_L, new_R)
                    if new_edges == orig_edges:
                        new_bics = list(bics)
                        new_bics[i] = [new_L, new_R]
                        return new_bics
        return None

    def try_reassign(bics, i, j):
        """Try to move a vertex from biclique i's L to biclique j's L (or R to R),
        if it maintains validity of both and edge partition."""
        L1, R1 = bics[i]
        L2, R2 = bics[j]
        # Try moving a vertex from L1 to L2
        for idx, v in enumerate(L1):
            new_L1 = L1[:idx] + L1[idx+1:]
            new_L2 = L2 + [v]
            if not new_L1 or not new_L2:
                continue
            if is_valid_biclique(new_L1, R1) and is_valid_biclique(new_L2, R2):
                # Check edge partition still holds
                orig_e1 = biclique_edges_set(L1, R1)
                orig_e2 = biclique_edges_set(L2, R2)
                new_e1 = biclique_edges_set(new_L1, R1)
                new_e2 = biclique_edges_set(new_L2, R2)
                if orig_e1 | orig_e2 == new_e1 | new_e2 and not (new_e1 & new_e2):
                    new_bics = list(bics)
                    new_bics[i] = [new_L1, R1]
                    new_bics[j] = [new_L2, R2]
                    return new_bics
        # Try moving a vertex from R1 to R2
        for idx, v in enumerate(R1):
            new_R1 = R1[:idx] + R1[idx+1:]
            new_R2 = R2 + [v]
            if not new_R1 or not new_R2:
                continue
            if is_valid_biclique(L1, new_R1) and is_valid_biclique(L2, new_R2):
                orig_e1 = biclique_edges_set(L1, R1)
                orig_e2 = biclique_edges_set(L2, R2)
                new_e1 = biclique_edges_set(L1, new_R1)
                new_e2 = biclique_edges_set(L2, new_R2)
                if orig_e1 | orig_e2 == new_e1 | new_e2 and not (new_e1 & new_e2):
                    new_bics = list(bics)
                    new_bics[i] = [L1, new_R1]
                    new_bics[j] = [L2, new_R2]
                    return new_bics
        return None

    def sa_search(bics, time_limit):
        """Simulated annealing to minimize number of bicliques."""
        best = bics
        best_count = len(bics)
        current = [b[:] for b in bics]  # deep copy
        current_count = len(current)
        T = 1.0
        T_min = 0.001
        cooling = 0.995

        while time.time() < time_limit and T > T_min:
            move_type = random.random()
            new_state = None

            if move_type < 0.5 and len(current) >= 2:
                # Merge move
                i, j = random.sample(range(len(current)), 2)
                new_state = try_merge(current, i, j)
                if new_state is not None:
                    new_count = len(new_state)
                    delta = new_count - current_count
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current = new_state
                        current_count = new_count
                        if current_count < best_count:
                            best = [b[:] for b in current]
                            best_count = current_count
            elif move_type < 0.75:
                # Split move (increases count, helps escape)
                i = random.randint(0, len(current) - 1)
                new_state = try_split(current, i)
                if new_state is not None:
                    new_count = len(new_state)
                    delta = new_count - current_count
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current = new_state
                        current_count = new_count
                        if current_count < best_count:
                            best = [b[:] for b in current]
                            best_count = current_count
            elif move_type < 0.95 and len(current) >= 2:
                # Reassign move
                i, j = random.sample(range(len(current)), 2)
                new_state = try_reassign(current, i, j)
                if new_state is not None:
                    new_count = len(new_state)
                    delta = new_count - current_count
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current = new_state
                        current_count = new_count
                        if current_count < best_count:
                            best = [b[:] for b in current]
                            best_count = current_count
            else:
                # Random restart with small perturbation: pick a biclique and swap a vertex
                i = random.randint(0, len(current) - 1)
                L, R = current[i]
                if len(L) >= 2:
                    idx = random.randint(0, len(L) - 1)
                    v = L[idx]
                    # Move v from L to R
                    new_L = L[:idx] + L[idx+1:]
                    new_R = R + [v]
                    if new_L and is_valid_biclique(new_L, new_R):
                        new_state = list(current)
                        new_state[i] = [new_L, new_R]
                        # Validate full partition
                        valid, cnt, _ = validate_and_count(new_state)
                        if valid:
                            delta = cnt - current_count
                            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                                current = new_state
                                current_count = cnt
                                if cnt < best_count:
                                    best = [b[:] for b in current]
                                    best_count = cnt

            T *= cooling

        return best, best_count

    # Main
    random.seed(42)

    # Get initial solution
    current = greedy_init()
    valid, cnt, _ = validate_and_count(current)
    if not valid:
        # Fallback: each edge as its own biclique
        current = [[u, v] for u, v in edges]
        cnt = len(current)

    # Run SA with multiple restarts
    best_bics = current
    best_cnt = cnt

    while time.time() < deadline:
        # Perturb: random restart from a modified initial
        if time.time() > start_time + 2:
            # Occasionally re-initialize
            perturbed = greedy_init()
            valid, cnt2, _ = validate_and_count(perturbed)
            if valid and cnt2 < best_cnt:
                best_bics = perturbed
                best_cnt = cnt2
            current = perturbed if valid else best_bics
        else:
            current = [b[:] for b in best_bics]

        # Run SA for a fixed number of iterations or time slice
        sa_time_limit = min(time.time() + 15, deadline - 2)
        result, res_cnt = sa_search(current, sa_time_limit)
        if res_cnt < best_cnt:
            best_bics = result
            best_cnt = res_cnt

        # Check time
        if time.time() > deadline - 3:
            break

    # Final validation
    valid, final_cnt, _ = validate_and_count(best_bics)
    if not valid:
        # Fallback to initial
        valid, final_cnt, _ = validate_and_count(current)
        if not valid:
            return []

    # Type conversion
    final_result = []
    for L, R in best_bics:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])

    return final_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
