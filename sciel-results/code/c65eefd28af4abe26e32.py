import time
import random
import numpy as np
from collections import defaultdict
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_index_map(edges):
        m = len(edges)
        idx = {}
        for i, (u, v) in enumerate(edges):
            idx[(u, v)] = i
            idx[(v, u)] = i
        return idx, m

    def clique_edges(clique, eidx):
        """Return list of edge indices in the clique."""
        res = []
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = clique[i], clique[j]
                e = eidx.get((a, b))
                if e is None:
                    return None
                res.append(e)
        return res

    def find_all_cliques(adj, n, max_size):
        """Find all cliques of size >=2 up to max_size using Bron-Kerbosch on complement-ish expansion."""
        # We'll enumerate maximal cliques first, then take all sub-cliques of size >=2? 
        # That's too many. Instead, enumerate all cliques of size 2..max_size directly.
        cliques = []
        # Use recursive expansion: for each vertex, find cliques containing it with higher-indexed vertices
        # This avoids duplicates by ordering.
        def expand(start, current, adj, n):
            if len(current) >= 2:
                cliques.append(list(current))
            if len(current) >= max_size:
                return
            # Candidates: vertices > last in current that are adjacent to all in current
            last = current[-1]
            candidates = [v for v in range(last + 1, n) if all(v in adj[x] for x in current)]
            for v in candidates:
                expand(start, current + [v], adj, n)
        
        for v in range(n):
            expand(v, [v], adj, n)
        return cliques

    def solve_lp(columns, m):
        """
        columns: list of lists of edge indices (each column is a clique's edge set)
        m: number of edges
        Solve min sum x_j s.t. sum_j A[i,j] x_j = 1 for all i, x_j >= 0
        """
        if not columns:
            return None, None
        ncol = len(columns)
        # A_eq: m x ncol
        A_eq = np.zeros((m, ncol))
        for j, col in enumerate(columns):
            for i in col:
                A_eq[i, j] = 1.0
        c = np.ones(ncol)
        res = linprog(c, A_eq=A_eq, b=np.ones(m), bounds=(0, None), method='highs')
        if res.success:
            return res.x, res.fun
        return None, None

    def pricing(columns, adj, n, m, eidx, duals):
        """
        Find a clique with negative reduced cost.
        Reduced cost of clique C = 1 - sum_{e in C} duals[e]
        We want to find C minimizing (1 - sum duals) = maximizing (sum duals - 1)
        This is a weighted clique problem with weights = duals[e].
        Use greedy heuristic: repeatedly add vertex that maximizes marginal gain.
        """
        # Weighted clique: weight of edge (u,v) = duals[eidx[(u,v)]]
        # Find clique maximizing sum of edge weights - 1 > 0
        best_gain = 0
        best_clique = None
        
        # Greedy: start from each edge, try to expand
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                e_uv = eidx[(u, v)]
                gain = duals[e_uv]
                clique = [u, v]
                # Try to add vertices
                changed = True
                while changed and len(clique) < n:
                    changed = False
                    # Candidates adjacent to all in clique
                    candidates = set(adj[clique[0]])
                    for x in clique[1:]:
                        candidates &= adj[x]
                    candidates = {x for x in candidates if x not in clique}
                    for w in candidates:
                        # Marginal gain of adding w
                        marg = 0
                        for x in clique:
                            e = eidx.get((min(x, w), max(x, w)))
                            if e is not None:
                                marg += duals[e]
                        if marg > 0:
                            # Add the one with highest marginal gain
                            if not changed or marg > best_marg:
                                best_marg = marg
                                best_w = w
                            changed = True
                    if changed:
                        clique.append(best_w)
                        gain += best_marg
                if gain - 1 > best_gain:
                    best_gain = gain - 1
                    best_clique = list(clique)
        
        if best_clique is not None and best_gain > 0:
            return best_clique, best_gain
        return None, 0

    def round_and_repair(lp_solution, columns, edges, adj, n, eidx, m):
        """
        Round LP solution to integer partition.
        Use greedy: pick columns with high lp value, ensure edge coverage.
        """
        if lp_solution is None:
            # Fallback: each edge as its own clique
            return [[edges[i][0], edges[i][1]] for i in range(m)]
        
        # Sort columns by lp value descending
        order = np.argsort(lp_solution)[::-1]
        covered = [0] * m
        selected = []
        
        for j in order:
            if lp_solution[j] < 0.01:
                break
            col = columns[j]
            # Check if adding this column would over-cover any edge
            can_add = True
            for i in col:
                if covered[i] >= 1:
                    can_add = False
                    break
            if can_add:
                selected.append(j)
                for i in col:
                    covered[i] += 1
        
        # Handle uncovered edges: add them as 2-cliques
        for i in range(m):
            if covered[i] == 0:
                u, v = edges[i]
                selected.append(len(columns))  # placeholder
                # We'll handle these separately
        
        # Build actual partition
        partition = []
        covered = [0] * m
        for j in selected:
            if j < len(columns):
                col = columns[j]
                # Check no over-coverage
                ok = True
                for i in col:
                    if covered[i] >= 1:
                        ok = False
                        break
                if ok:
                    # Get the clique vertices
                    clique = None
                    for c in columns:
                        if c == col:
                            clique = c
                            break
                    if clique is None:
                        # Reconstruct from col edges
                        verts = set()
                        for i in col:
                            u, v = edges[i]
                            verts.add(u)
                            verts.add(v)
                        clique = list(verts)
                    partition.append(clique)
                    for i in col:
                        covered[i] = 1
        
        # Add uncovered edges as 2-cliques
        for i in range(m):
            if covered[i] == 0:
                u, v = edges[i]
                partition.append([u, v])
                covered[i] = 1
        
        return partition

    def validate_partition(cliques, edges, adj, n, eidx, m):
        """Check if partition is valid."""
        edge_count = [0] * m
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check clique property
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = eidx.get((a, b))
                    if e is None:
                        return False
                    edge_count[e] += 1
        for i in range(m):
            if edge_count[i] != 1:
                return False
        return True

    def local_search(partition, edges, adj, n, eidx, m, time_budget):
        """
        Local search: try to merge two cliques if their union is a clique and no edge conflict.
        Try to split a clique into two smaller cliques if it reduces count (never does for count, but may enable merges).
        Actually, since we minimize count, only merges help. But merges require the union to be a clique.
        """
        current = partition[:]
        current_count = len(current)
        best = current[:]
        best_count = current_count
        
        rng = random.Random(123)
        
        # Precompute edge sets for each clique
        def get_clique_edge_set(clique):
            s = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = eidx.get((a, b))
                    if e is not None:
                        s.add(e)
            return s
        
        # Map edge -> clique index
        def build_edge_map(cliques):
            emap = {}
            for i, clique in enumerate(cliques):
                for e in get_clique_edge_set(clique):
                    emap[e] = i
            return emap
        
        emap = build_edge_map(current)
        
        while time.time() < time_budget and current_count > 1:
            # Try to find two cliques whose union is a clique
            merged = False
            for i in range(len(current)):
                if time.time() >= time_budget:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() >= time_budget:
                        break
                    c1 = current[i]
                    c2 = current[j]
                    union = list(set(c1) | set(c2))
                    if len(union) < 2:
                        continue
                    # Check if union is a clique
                    is_clique = True
                    for a in union:
                        for b in union:
                            if a < b:
                                if b not in adj[a]:
                                    is_clique = False
                                    break
                        if not is_clique:
                            break
                    if not is_clique:
                        continue
                    # Check edge conflict: no edge in union should be in other cliques
                    union_edges = set()
                    for a in union:
                        for b in union:
                            if a < b:
                                e = eidx.get((a, b))
                                if e is not None:
                                    union_edges.add(e)
                    # Check if any edge in union_edges is covered by a clique other than i or j
                    conflict = False
                    for e in union_edges:
                        if e in emap and emap[e] not in (i, j):
                            conflict = True
                            break
                    if conflict:
                        continue
                    # Perform merge
                    new_current = current[:]
                    del new_current[max(i, j)]
                    del new_current[min(i, j)]
                    new_current.append(union)
                    new_count = len(new_current)
                    if new_count < best_count:
                        current = new_current
                        current_count = new_count
                        best = current[:]
                        best_count = current_count
                        emap = build_edge_map(current)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                # No more merges possible, try random perturbation
                if rng.random() < 0.3:
                    # Try splitting a large clique and re-merging differently
                    large_idx = -1
                    for i, c in enumerate(current):
                        if len(c) >= 4:
                            large_idx = i
                            break
                    if large_idx >= 0:
                        c = current[large_idx]
                        # Split into two parts
                        mid = len(c) // 2
                        part1 = c[:mid]
                        part2 = c[mid:]
                        if len(part1) >= 2 and len(part2) >= 2:
                            # Check both are cliques
                            if all(all(y in adj[x] for y in part1) for x in part1) and \
                               all(all(y in adj[x] for y in part2) for x in part2):
                                new_current = current[:]
                                new_current[large_idx] = part1
                                new_current.append(part2)
                                # This increases count, but may enable better merges later
                                # Only accept if it leads to improvement within a few steps
                                # For simplicity, skip this for now
                                pass
                break
        
        return best

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_graph(n, p, seed)
        adj = build_adj(edges, n)
        eidx, m = edge_index_map(edges)
        
        # Find all cliques of size up to 6 (dense graphs, many small cliques)
        # For n=32, p=0.6: expected clique size is small
        # For n=64, p=0.55: even smaller
        max_clique_size = 5
        all_cliques = find_all_cliques(adj, n, max_clique_size)
        
        # Convert to edge lists
        columns = []
        for clique in all_cliques:
            col = clique_edges(clique, eidx)
            if col is not None and len(col) > 0:
                columns.append(col)
        
        # Add each edge as a column (2-clique)
        for i in range(m):
            columns.append([i])
        
        # Solve LP
        lp_sol, lp_obj = solve_lp(columns, m)
        
        # Round and repair
        partition = round_and_repair(lp_sol, columns, edges, adj, n, eidx, m)
        
        # Validate
        if not validate_partition(partition, edges, adj, n, eidx, m):
            partition = [[edges[i][0], edges[i][1]] for i in range(m)]
        
        # Local search
        time_budget = min(deadline - 5.0, time.time() + 30.0)
        partition = local_search(partition, edges, adj, n, eidx, m, time_budget)
        
        # Final validation
        if not validate_partition(partition, edges, adj, n, eidx, m):
            partition = [[edges[i][0], edges[i][1]] for i in range(m)]
        
        results.append(partition)
    
    return results
