import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng):
        """
        Finds a long cycle in the graph using iterative DFS with proper cycle extraction.
        """
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_len = 0

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        starts = candidates[:min(8, len(candidates))]
        if len(candidates) > 8:
            starts += rng.sample(candidates, min(8, len(candidates) - 8))
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            color = {}
            parent = {}
            path = []
            path_index = {}
            local_best_cycle = None
            local_best_len = 0

            # Iterative DFS
            stack = [(start, iter(adj[start]))]
            color[start] = 1
            parent[start] = -1
            path.append(start)
            path_index[start] = 0

            while stack:
                u, neighbors_iter = stack[-1]
                found_next = False
                for w in neighbors_iter:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        # Back edge: w is on current path
                        idx = path_index.get(w)
                        if idx is not None:
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if path_index:
                            del path_index[u]

            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
            if best_len >= 25:
                break

        return best_cycle

    def greedy_decompose(edges, rng):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_adj = {u: set(v) for u, v in adj.items()}
        current_edges = set(edges)
        pieces = []
        
        while True:
            if time.time() - start_time > DEADLINE - 2:
                break
            if not any(len(neigh) > 0 for neigh in current_adj.values()):
                break
            
            cycle = find_best_cycle(current_adj, rng)
            if cycle is None:
                break
            
            for j in range(len(cycle)):
                a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                current_edges.discard(e)
                if b in current_adj[a]:
                    current_adj[a].discard(b)
                if a in current_adj[b]:
                    current_adj[b].discard(a)
            
            pieces.append(list(cycle))
        
        for e in current_edges:
            pieces.append([e[0], e[1]])
            
        return pieces

    def solve_with_ilp(edges, candidate_cycles, time_limit):
        # candidate_cycles: list of (vertex_tuple, edge_frozenset)
        if not candidate_cycles:
            return None
            
        E = len(edges)
        M = len(candidate_cycles)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build sparse matrix A (E x (M+E))
        # A[e_idx, var_idx] = 1 if var covers edge e
        rows = []
        cols = []
        data = []
        
        for i, (verts, e_set) in enumerate(candidate_cycles):
            for e in e_set:
                if e in edge_to_idx:
                    rows.append(edge_to_idx[e])
                    cols.append(i)
                    data.append(1)
                    
        for j in range(E):
            rows.append(j)
            cols.append(M + j)
            data.append(1)
            
        A = csr_matrix((data, (rows, cols)), shape=(E, M + E))
        
        c = np.ones(M + E)
        
        # Constraints: A x = 1
        constraints = LinearConstraint(A, np.ones(E), np.ones(E))
        
        integrality = np.ones(M + E)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0: # Optimal
                x = res.x
                pieces = []
                for i in range(M):
                    if x[i] > 0.5:
                        # Add cycle
                        pieces.append(list(candidate_cycles[i][0]))
                for j in range(E):
                    if x[M+j] > 0.5:
                        # Add single edge
                        pieces.append([edges[j][0], edges[j][1]])
                return pieces
            else:
                return None
        except Exception:
            return None

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        
        # 1. Generate candidate cycles
        cycle_dict = {} # edge_frozenset -> vertex_list
        best_greedy_pieces = None
        best_greedy_count = float('inf')
        
        # Run greedy multiple times
        max_restarts = 100
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE - 15:
                break
            rng = random.Random(i * 7 + 3)
            pieces = greedy_decompose(edges, rng)
            
            # Extract cycles
            for p in pieces:
                if len(p) > 2:
                    # Ensure simple cycle representation
                    # p is a list of vertices
                    cycle_edges = frozenset((min(u,v), max(u,v)) for u,v in zip(p, p[1:]+p[:1]))
                    if cycle_edges not in cycle_dict:
                        cycle_dict[cycle_edges] = p
            
            count = len(pieces)
            if count < best_greedy_count:
                best_greedy_count = count
                best_greedy_pieces = [list(p) for p in pieces]
                
        # 2. Try ILP
        if time.time() - start_time < DEADLINE - 5 and len(cycle_dict) > 0:
            # Convert dict to list of (verts, edges)
            candidate_list = [(verts, edges_set) for edges_set, verts in cycle_dict.items()]
            
            # Sort by length descending to prioritize long cycles in the pool if we need to truncate
            candidate_list.sort(key=lambda x: len(x[0]), reverse=True)
            
            # Use all candidates, or truncate if too many?
            # For N=30, M can be a few hundred. ILP handles this fine.
            # Let's cap at 500 to be safe.
            if len(candidate_list) > 500:
                candidate_list = candidate_list[:500]
                
            ilp_pieces = solve_with_ilp(edges, candidate_list, DEADLINE - time.time() - 5)
            
            if ilp_pieces is not None:
                if len(ilp_pieces) < best_greedy_count:
                    best_greedy_pieces = ilp_pieces
                    best_greedy_count = len(ilp_pieces)
                    
        if best_greedy_pieces is None:
            best_greedy_pieces = [[u, v] for u, v in edges]
            
        results.append(best_greedy_pieces)
        
    return results
