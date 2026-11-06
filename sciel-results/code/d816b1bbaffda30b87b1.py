import time
import sys
from collections import defaultdict
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_graph_info(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_all_simple_paths(adj, n, max_len=None):
        """
        Find all simple paths in the graph.
        For sparse graphs with n<=30, the number of simple paths can be large, 
        but we can limit length or use a heuristic to prune.
        However, enumerating ALL simple paths is exponential. 
        We need a smarter way to generate candidate paths or use a different formulation.
        
        Alternative Formulation for MILP:
        Instead of enumerating all paths (which is too many), we can model the flow directly.
        Variables:
        x_e: 1 if edge e is used in a path (must be 1 for all edges)
        This doesn't minimize paths directly.
        
        Let's use the "Path Selection" formulation but restrict candidates.
        Actually, for n=30, the number of simple paths is huge. 
        However, we can use a "Labeling" approach or just rely on the fact that 
        optimal paths in sparse graphs are often long.
        
        A better structural change: Use the MILP to select a subset of edges to form paths?
        No, all edges must be covered.
        
        Let's try the "Flow" formulation for path decomposition.
        Minimize sum of start nodes (or number of paths).
        Constraints:
        1. Flow conservation at each node for each "commodity" (path). 
           But we don't know the number of paths.
        
        Standard formulation for Edge-Disjoint Path Decomposition:
        Let P be the set of all simple paths in G.
        Variable y_p in {0,1} for each path p in P.
        Minimize sum y_p
        Subject to:
        sum_{p in P, e in p} y_p = 1 for all e in E
        y_p >= 0
        
        The bottleneck is generating P.
        For n=30, we cannot generate all simple paths.
        
        Heuristic for P generation:
        Generate a pool of "good" paths (long ones) using a randomized greedy algorithm, 
        then solve the MILP on this pool. If the MILP solution is not integral or doesn't cover all edges, 
        we might need to add more paths (cutting planes) or just accept the greedy result if MILP is too slow.
        
        Given the time limit (150s), generating a large pool of paths might be risky.
        Let's generate ~200-500 random long paths for each graph.
        """
        paths = []
        seen = set()
        
        # Greedy generation of long simple paths
        def generate_random_path(adj, n):
            # Pick a random start node
            start = random.randint(0, n-1)
            path = [start]
            used_edges = set()
            current = start
            
            while True:
                neighbors = list(adj[current])
                random.shuffle(neighbors)
                next_node = None
                for nb in neighbors:
                    e = tuple(sorted((current, nb)))
                    if e not in used_edges and nb not in path:
                        next_node = nb
                        used_edges.add(e)
                        break
                if next_node is None:
                    break
                path.append(next_node)
                current = next_node
                
            if len(path) >= 2:
                # Canonical representation of the path's edges
                edge_set = frozenset(tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1))
                if edge_set not in seen:
                    seen.add(edge_set)
                    return path
            return None

        # Generate a pool of paths
        # We want long paths to minimize the count.
        # Try multiple times to find long paths.
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            p = generate_random_path(adj, n)
            if p:
                paths.append(p)
                
        return paths

    def solve_milp(edges, n, adj, paths_pool):
        """
        Solve the set cover problem:
        Minimize sum(y_p)
        s.t. sum(y_p for p in pool if e in p) = 1 for all e in edges
        y_p in {0,1}
        """
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        m = len(edges)
        num_paths = len(paths_pool)
        
        if num_paths == 0:
            # Fallback: single edges
            return [[u, v] for u, v in edges]

        # Build incidence matrix A (m x num_paths)
        # A[i, j] = 1 if edge i is in path j
        edge_index = {e: i for i, e in enumerate(edges)}
        
        A = np.zeros((m, num_paths))
        for j, path in enumerate(paths_pool):
            for k in range(len(path) - 1):
                e = tuple(sorted((path[k], path[k+1])))
                if e in edge_index:
                    A[edge_index[e], j] = 1
                else:
                    # This shouldn't happen if paths are valid
                    pass

        c = np.ones(num_paths)
        
        # Constraints: A @ y = 1
        # 1 <= (A @ y) <= 1
        constraint = LinearConstraint(A, lb=1, ub=1)
        integrality = np.ones(num_paths)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                selected_paths = [paths_pool[i] for i in selected_indices]
                return selected_paths
            else:
                # Infeasible or failed. Fallback to greedy
                return None
        except Exception:
            return None

    def greedy_decompose(edges, n, adj):
        """Fallback greedy decomposition"""
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining_edges:
            # Pick an edge
            e = next(iter(remaining_edges))
            u, v = e
            # Extend from u
            path = [u]
            used_edges = {e}
            current = u
            # Try to extend backwards from u (conceptually, but we start with u-v)
            # Let's just do a simple DFS extension from u and v
            # Actually, let's do a simple walk
            # Start at u, go to v
            path.append(v)
            current = v
            
            while True:
                extended = False
                for nb in adj[current]:
                    ne = tuple(sorted((current, nb)))
                    if ne in remaining_edges and nb not in path:
                        path.append(nb)
                        remaining_edges.discard(ne)
                        current = nb
                        extended = True
                        break
                if not extended:
                    break
            
            # Remove the initial edge from remaining (it was added to used but not discarded from remaining in the loop above if we didn't start with it? 
            # Wait, I initialized used_edges but didn't use it to filter remaining_edges properly in the extension loop.
            # The extension loop checks `if ne in remaining_edges`. 
            # The first edge e was not removed from remaining_edges before the loop.
            # So the loop might pick e again? No, e is (u,v). current is v. nb would be u. (v,u) is e. 
            # But u is in path. So it won't pick it.
            # But e is still in remaining_edges.
            remaining_edges.discard(e)
            
            paths.append(path)
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            break
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_graph_info(edges, n)
        
        # Try MILP approach
        # Generate pool
        random.seed(seed) # Deterministic pool generation
        paths_pool = find_all_simple_paths(adj, n)
        
        best_paths = None
        if paths_pool:
            milp_result = solve_milp(edges, n, adj, paths_pool)
            if milp_result is not None:
                # Verify validity
                valid = True
                used_edges = set()
                for path in milp_result:
                    for i in range(len(path)-1):
                        e = tuple(sorted((path[i], path[i+1])))
                        if e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid: break
                if valid and used_edges == set(tuple(sorted(e)) for e in edges):
                    best_paths = milp_result
        
        if best_paths is None:
            # Fallback to greedy
            best_paths = greedy_decompose(edges, n, adj)
            
        results.append(best_paths)
        
    # Fill remaining with trivial if time ran out
    while len(results) < 3:
        spec = specs[len(results)]
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
        
    return results
