import time
import random
from collections import defaultdict
from itertools import combinations
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute edge to candidates
        k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)
        
        # Sort edges by scarcity (number of candidates)
        edges_by_scarcity = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
        
        best_result = None
        best_count = float('inf')
        
        max_iters = 100
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            used_edges = set()
            v_matched = {v: set() for v in range(k, n)}
            v_cliques = {v: [] for v in range(k, n)}
            
            # Create a list of edges to process
            edges_to_process = edges_by_scarcity[:]
            random.shuffle(edges_to_process)
            edges_to_process.sort(key=lambda e: len(edge_to_vs[e]))
            
            for e in edges_to_process:
                u, w = e
                # Find valid candidates
                candidates = [v for v in edge_to_vs[e] 
                              if u not in v_matched[v] and w not in v_matched[v]]
                if not candidates:
                    continue
                
                # Pick best candidate
                if random.random() < 0.1:
                    v = random.choice(candidates)
                else:
                    v = max(candidates, key=lambda v: len(set(neighbors[v]) - v_matched[v]))
                
                # Assign
                v_matched[v].add(u)
                v_matched[v].add(w)
                v_cliques[v].append([u, w, v])
                used_edges.add(e)
            
            # Build cliques
            cliques = []
            
            # 3-cliques from matching
            for v in range(k, n):
                cliques.extend(v_cliques[v])
            
            # 2-cliques for unmatched neighbors
            for v in range(k, n):
                N_v = set(neighbors[v])
                matched = v_matched[v]
                for u in N_v - matched:
                    cliques.append([u, v])
            
            # Cover remaining K-edges
            remaining_edges = set(k_edges) - used_edges
            rem_set = set(remaining_edges)
            
            while rem_set:
                # Find vertex with max degree in rem_set
                deg = defaultdict(int)
                for u, w in rem_set:
                    deg[u] += 1
                    deg[w] += 1
                
                if not deg:
                    break
                    
                max_deg_vertex = max(deg, key=deg.get)
                
                neighbors_in_rem = set()
                for u, w in rem_set:
                    if u == max_deg_vertex:
                        neighbors_in_rem.add(w)
                    elif w == max_deg_vertex:
                        neighbors_in_rem.add(u)
                
                clique = {max_deg_vertex}
                changed = True
                while changed:
                    changed = False
                    for x in list(neighbors_in_rem):
                        if x in clique:
                            continue
                        is_connected = True
                        for y in clique:
                            edge = (min(x, y), max(x, y))
                            if edge not in rem_set:
                                is_connected = False
                                break
                        if is_connected:
                            clique.add(x)
                            changed = True
                
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    for a, b in combinations(sorted(clique), 2):
                        rem_set.discard((a, b))
                else:
                    e = next(iter(rem_set))
                    cliques.append(list(e))
                    rem_set.discard(e)
            
            # Validate
            clean = []
            for c in cliques:
                clean.append([int(x) for x in c])
            
            # Verify
            edge_count = defaultdict(int)
            for c in clean:
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        u, w = min(c[i], c[j]), max(c[i], c[j])
                        edge_count[(u, w)] += 1
            
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            valid = True
            for e in all_edges:
                if edge_count[e] != 1:
                    valid = False
                    break
            for e, cnt in edge_count.items():
                if e not in all_edges:
                    valid = False
                    break
            
            if not valid:
                continue
            
            count = len(clean)
            if count < best_count:
                best_count = count
                best_result = clean
                
        # Try Hungarian algorithm for optimal matching
        if time.time() < deadline - 10:
            try:
                # Build cost matrix for Hungarian algorithm
                # Rows: edges in K, Cols: vertices in I
                num_edges = len(k_edges)
                num_vertices = n - k
                
                # Cost matrix: cost[i][j] = 0 if vertex j can cover edge i, else large
                INF_COST = 10**9
                cost_matrix = [[INF_COST] * num_vertices for _ in range(num_edges)]
                
                for i, e in enumerate(k_edges):
                    u, w = e
                    for j, v in enumerate(range(k, n)):
                        if u in neighbors[v] and w in neighbors[v]:
                            cost_matrix[i][j] = 0
                
                # Solve assignment problem
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Build matching
                matching = {}
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r][c] == 0:
                        matching[k_edges[r]] = k + c
                
                # Build cliques from matching
                cliques = []
                v_matched = {v: set() for v in range(k, n)}
                
                for e, v in matching.items():
                    u, w = e
                    cliques.append([u, w, v])
                    v_matched[v].add(u)
                    v_matched[v].add(w)
                
                # 2-cliques for unmatched neighbors
                for v in range(k, n):
                    N_v = set(neighbors[v])
                    matched = v_matched[v]
                    for u in N_v - matched:
                        cliques.append([u, v])
                
                # Cover remaining K-edges
                used_edges = set(matching.keys())
                rem_set = set(k_edges) - used_edges
                
                while rem_set:
                    deg = defaultdict(int)
                    for u, w in rem_set:
                        deg[u] += 1
                        deg[w] += 1
                    
                    if not deg:
                        break
                        
                    max_deg_vertex = max(deg, key=deg.get)
                    
                    neighbors_in_rem = set()
                    for u, w in rem_set:
                        if u == max_deg_vertex:
                            neighbors_in_rem.add(w)
                        elif w == max_deg_vertex:
                            neighbors_in_rem.add(u)
                    
                    clique = {max_deg_vertex}
                    changed = True
                    while changed:
                        changed = False
                        for x in list(neighbors_in_rem):
                            if x in clique:
                                continue
                            is_connected = True
                            for y in clique:
                                edge = (min(x, y), max(x, y))
                                if edge not in rem_set:
                                    is_connected = False
                                    break
                            if is_connected:
                                clique.add(x)
                                changed = True
                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(sorted(clique), 2):
                            rem_set.discard((a, b))
                    else:
                        e = next(iter(rem_set))
                        cliques.append(list(e))
                        rem_set.discard(e)
                
                # Validate
                clean = []
                for c in cliques:
                    clean.append([int(x) for x in c])
                
                # Verify
                edge_count = defaultdict(int)
                for c in clean:
                    for i in range(len(c)):
                        for j in range(i + 1, len(c)):
                            u, w = min(c[i], c[j]), max(c[i], c[j])
                            edge_count[(u, w)] += 1
                
                all_edges = set()
                for u in range(k):
                    for w in range(u + 1, k):
                        all_edges.add((u, w))
                for v in range(k, n):
                    for u in neighbors[v]:
                        all_edges.add((min(u, v), max(u, v)))
                
                valid = True
                for e in all_edges:
                    if edge_count[e] != 1:
                        valid = False
                        break
                for e, cnt in edge_count.items():
                    if e not in all_edges:
                        valid = False
                        break
                
                if valid:
                    count = len(clean)
                    if count < best_count:
                        best_count = count
                        best_result = clean
                        
            except Exception:
                pass
        
        if best_result is None:
            # Fallback
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            best_result = [[int(x) for x in c] for c in cliques]
            
        return best_result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    while len(results) < 3:
        results.append([])
        
    return results
