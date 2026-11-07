"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds, leave safety margin

    n = P['n']

    # Build initial adjacency from path
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    def bfs_all_pairs(adj_mat):
        """Compute all-pairs shortest paths using BFS from each node."""
        dist_matrix = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj_mat[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            dist_matrix.append(dist)
        return dist_matrix

    def get_diameter(dist_matrix):
        """Get diameter from distance matrix."""
        max_diam = 0
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][j] > max_diam:
                    max_diam = dist_matrix[i][j]
        return max_diam

    # Hardcoded solution from the seed program (8 edges)
    hardcoded_edges = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]

    def check_valid(added_edges):
        """Verify the full set of added edges is valid."""
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if adj[a][b]:
                return False

        tmp_adj = [row[:] for row in adj]
        for u, v in added_edges:
            tmp_adj[u][v] = True
            tmp_adj[v][u] = True

        # Check triangle-free
        for u in range(n):
            for v in range(u+1, n):
                if tmp_adj[u][v]:
                    for w in range(n):
                        if w != u and w != v and tmp_adj[u][w] and tmp_adj[v][w]:
                            return False

        dist_matrix = bfs_all_pairs(tmp_adj)
        return get_diameter(dist_matrix) <= 4

    # First, try the hardcoded solution
    if check_valid(hardcoded_edges):
        # Try to do better with ILP
        pass

    # Generate all candidate edges (non-existing, triangle-free with path)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                # Check triangle-free with just the path
                triangle_free = True
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        triangle_free = False
                        break
                if triangle_free:
                    candidates.append((u, v))

    m = len(candidates)

    if m == 0:
        if check_valid(hardcoded_edges):
            return hardcoded_edges
        return []

    # Compute distances in base graph
    base_dist = bfs_all_pairs(adj)

    # For each candidate edge (u,v), compute the new distance contribution
    # When we add edge (u,v), for any pair (i,j), new_dist(i,j) <= min(old_dist(i,j),
    #   dist(i,u)+1+dist(v,j), dist(i,v)+1+dist(u,j))
    # We need new_dist(i,j) <= 4 for all i,j

    # Precompute for each candidate edge e=(u,v) and each pair (i,j):
    # the "improvement" it provides: can it bring dist(i,j) down to <= 4?
    # This is a linear constraint formulation.

    # ILP formulation:
    # Variables: x_e in {0,1} for each candidate edge e
    # Minimize: sum x_e
    # Subject to:
    #   For each pair (i,j) with base_dist(i,j) > 4:
    #     sum over edges e that can cover (i,j) of x_e >= 1
    #     where "can cover" means adding e makes dist(i,j) <= 4
    #   Triangle-free constraints:
    #     For each triangle (u,v,w) in the complete graph on candidates:
    #       x_(u,v) + x_(v,w) + x_(u,w) <= 2
    #     (i.e., can't have all three edges of a triangle)

    # Determine which pairs need coverage
    pairs_needing = []
    for i in range(n):
        for j in range(i + 1, n):
            if base_dist[i][j] > 4:
                pairs_needing.append((i, j))

    if not pairs_needing:
        return []

    # For each candidate edge, determine which pairs it covers
    # Edge (u,v) covers pair (i,j) if:
    #   min(base_dist[i][j], base_dist[i][u]+1+base_dist[v][j], base_dist[i][v]+1+base_dist[u][j]) <= 4
    edge_covers = []
    for idx, (u, v) in enumerate(candidates):
        covers = []
        for pi, (i, j) in enumerate(pairs_needing):
            d1 = base_dist[i][u] + 1 + base_dist[v][j] if base_dist[i][u] != -1 and base_dist[v][j] != -1 else float('inf')
            d2 = base_dist[i][v] + 1 + base_dist[u][j] if base_dist[i][v] != -1 and base_dist[u][j] != -1 else float('inf')
            if min(base_dist[i][j], d1, d2) <= 4:
                covers.append(pi)
        edge_covers.append(covers)

    # Check if any single edge covers all pairs
    # Also check if a small set works

    # Triangle-free constraints: for each triple of candidate edges that form a triangle
    # Edge a=(u1,v1), b=(u2,v2), c=(u3,v3) form a triangle if they share vertices appropriately
    # Actually, triangle-free means: for any three vertices u,v,w, we can't have all three edges (u,v),(v,w),(u,w)
    # So for each triple of vertices (u,v,w), if all three pairs are in candidates, add constraint x_(u,v)+x_(v,w)+x_(u,w) <= 2

    # Build constraint matrix for linprog
    # Objective: minimize sum(x_e)
    c_obj = np.ones(m)

    # Diameter constraints: for each pair (i,j) needing coverage, sum of x_e for e covering (i,j) >= 1
    # In linprog form: -sum(x_e) <= -1
    A_diam = []
    b_diam = []
    for pi, (i, j) in enumerate(pairs_needing):
        row = np.zeros(m)
        for ei in range(m):
            if pi in edge_covers[ei]:
                row[ei] = -1
        if row.sum() == 0:
            # No edge can cover this pair - infeasible
            if check_valid(hardcoded_edges):
                return hardcoded_edges
            return []
        A_diam.append(row)
        b_diam.append(-1)

    # Triangle-free constraints
    # For each triple of vertices (u,v,w) where all three pairs are candidates
    # Find indices of edges in candidates
    edge_to_idx = {}
    for idx, (u, v) in enumerate(candidates):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    A_tri = []
    b_tri = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                e1 = edge_to_idx.get((u, v))
                e2 = edge_to_idx.get((v, w))
                e3 = edge_to_idx.get((u, w))
                if e1 is not None and e2 is not None and e3 is not None:
                    row = np.zeros(m)
                    row[e1] = 1
                    row[e2] = 1
                    row[e3] = 1
                    A_tri.append(row)
                    b_tri.append(2)

    # Combine constraints
    if A_diam:
        A_ub = np.array(A_diam + A_tri) if A_tri else np.array(A_diam)
        b_ub = np.array(b_diam + b_tri) if A_tri else np.array(b_diam)
    else:
        A_ub = np.array(A_tri) if A_tri else None
        b_ub = np.array(b_tri) if A_tri else None

    # Bounds: 0 <= x_e <= 1
    bounds = [(0, 1)] * m

    # Solve LP relaxation first
    try:
        if A_ub is not None:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        else:
            res = linprog(c_obj, bounds=bounds, method='highs')

        if res.status == 0 and res.x is not None:
            # Round the LP solution
            # Use a greedy rounding approach
            x_lp = res.x
            # Sort by LP value descending, add edges with high values
            order = sorted(range(m), key=lambda i: -x_lp[i])
            selected = []
            selected_set = set()

            # Greedy: add edges in order of LP value, checking constraints
            for ei in order:
                if x_lp[ei] < 0.01:
                    continue
                u, v = candidates[ei]
                # Check triangle-free with already selected edges
                tri_ok = True
                for (su, sv) in selected_set:
                    # Check if adding (u,v) creates a triangle with any existing edge
                    # Triangle (u,v,w) exists if (u,w) and (v,w) are both in selected or base
                    for w in range(n):
                        if w != u and w != v:
                            if adj[u][w] and adj[v][w]:
                                tri_ok = False
                                break
                            if (u, w) in selected_set or (w, u) in selected_set:
                                if (v, w) in selected_set or (w, v) in selected_set:
                                    tri_ok = False
                                    break
                            if (v, w) in selected_set or (w, v) in selected_set:
                                if (u, w) in selected_set or (w, u) in selected_set:
                                    tri_ok = False
                                    break
                    if not tri_ok:
                        break
                if not tri_ok:
                    continue

                # Check if this helps with diameter
                selected.append([u, v])
                selected_set.add((u, v))
                selected_set.add((v, u))

                # Check if all pairs are now covered
                tmp_adj = [row[:] for row in adj]
                for eu, ev in selected:
                    tmp_adj[eu][ev] = True
                    tmp_adj[ev][eu] = True
                tmp_dist = bfs_all_pairs(tmp_adj)
                tmp_diam = get_diameter(tmp_dist)
                if tmp_diam <= 4:
                    break

                if time.time() > deadline - 5:
                    break

            if selected:
                # Verify
                if check_valid(selected):
                    # Try to improve: remove edges one by one
                    improved = True
                    while improved:
                        improved = False
                        for i in range(len(selected)):
                            trial = selected[:i] + selected[i+1:]
                            if check_valid(trial):
                                selected = trial
                                improved = True
                                break
                    return selected

    except Exception:
        pass

    # Fallback: greedy set cover
    current_adj = [row[:] for row in adj]
    current_dist = bfs_all_pairs(current_adj)
    current_diam = get_diameter(current_dist)

    if current_diam <= 4:
        return []

    def is_triangle_free_check(adj_mat, u, v):
        for w in range(n):
            if w != u and w != v and adj_mat[u][w] and adj_mat[v][w]:
                return False
        return True

    def edge_covers_pairs_fast(u, v, dist_matrix, target_diam):
        """Determine which pairs with dist > target_diam would be reduced by adding edge (u,v)."""
        covered = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i][j] <= target_diam:
                    continue
                d1 = dist_matrix[i][u] + 1 + dist_matrix[v][j] if dist_matrix[i][u] != -1 and dist_matrix[v][j] != -1 else float('inf')
                d2 = dist_matrix[i][v] + 1 + dist_matrix[u][j] if dist_matrix[i][v] != -1 and dist_matrix[u][j] != -1 else float('inf')
                new_dist = min(dist_matrix[i][j], d1, d2)
                if new_dist <= target_diam:
                    covered.append((i, j))
        return covered

    uncovered_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if current_dist[i][j] > 4:
                uncovered_pairs.add((i, j))

    selected_edges = []

    while uncovered_pairs and time.time() < deadline - 3:
        best_edge = None
        best_coverage = 0

        for u, v in candidates:
            # Check if already selected
            if (u, v) in selected_edges or (v, u) in selected_edges:
                continue
            if not is_triangle_free_check(current_adj, u, v):
                continue

            covered = edge_covers_pairs_fast(u, v, current_dist, 4)
            new_coverage = len([p for p in covered if p in uncovered_pairs])

            if new_coverage > best_coverage:
                best_coverage = new_coverage
                best_edge = (u, v)

        if best_edge is None or best_coverage == 0:
            break

        u, v = best_edge
        selected_edges.append([u, v])
        current_adj[u][v] = True
        current_adj[v][u] = True
        current_dist = bfs_all_pairs(current_adj)
        uncovered_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                if current_dist[i][j] > 4:
                    uncovered_pairs.add((i, j))

    # Verify the solution
    if not uncovered_pairs and check_valid(selected_edges):
        # Try to improve: remove edges one by one
        improved = True
        while improved:
            improved = False
            for i in range(len(selected_edges)):
                trial = selected_edges[:i] + selected_edges[i + 1:]
                if check_valid(trial):
                    selected_edges = trial
                    improved = True
                    break
        return selected_edges

    # Fallback to hardcoded
    if check_valid(hardcoded_edges):
        return hardcoded_edges

    return []

# EVOLVE-BLOCK-END
