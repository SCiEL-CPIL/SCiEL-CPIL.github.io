import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

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
        
        # List of all edges in K
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        
        # Map each edge to the set of vertices in I that can use it
        # edge -> list of v
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)
        
        # Greedy assignment of edges to vertices in I
        # We want to assign each edge to at most one v.
        # For each v, assigned edges must form a matching (disjoint vertices).
        
        # Sort edges by number of available v's (scarcity first)
        sorted_edges = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
        
        assigned_edges = {}  # v -> set of edges assigned to v
        for v in range(k, n):
            assigned_edges[v] = set()
            
        # Track which vertices in K are "used" by each v
        used_vertices_in_v = {v: set() for v in range(k, n)}
        
        best_cliques = None
        best_count = float('inf')
        
        # Try more random permutations/restarts
        num_restarts = 50
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            curr_assigned = {v: set() for v in range(k, n)}
            curr_used = {v: set() for v in range(k, n)}
            
            # Sort edges by scarcity
            edges_to_process = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
            
            # Randomize order
            random.shuffle(edges_to_process)
            
            for e in edges_to_process:
                candidates = edge_to_vs[e]
                if not candidates:
                    continue
                
                valid_vs = []
                for v in candidates:
                    u, w = e
                    if u not in curr_used[v] and w not in curr_used[v]:
                        valid_vs.append(v)
                
                if not valid_vs:
                    continue
                
                # Pick a v
                # Heuristic: Pick v with the most "free" capacity?
                # Or just random.
                v = random.choice(valid_vs)
                
                curr_assigned[v].add(e)
                curr_used[v].add(e[0])
                curr_used[v].add(e[1])
            
            # Calculate cliques for this assignment
            cliques = []
            
            # 1. Cliques from assigned edges
            for v in range(k, n):
                for e in curr_assigned[v]:
                    cliques.append([e[0], e[1], v])
            
            # 2. Cliques for remaining neighbors (singles)
            for v in range(k, n):
                N_v = set(neighbors[v])
                used = curr_used[v]
                remaining = N_v - used
                for u in remaining:
                    cliques.append([u, v])
            
            # 3. Cover remaining edges in K
            assigned_edge_set = set()
            for v in range(k, n):
                for e in curr_assigned[v]:
                    assigned_edge_set.add(e)
            
            remaining_k_edges = [e for e in k_edges if e not in assigned_edge_set]
            
            if remaining_k_edges:
                # Greedy clique cover for remaining K edges
                rem_edges_set = set(remaining_k_edges)
                while rem_edges_set:
                    if time.time() > deadline:
                        break
                    e = next(iter(rem_edges_set))
                    u, w = e
                    clique = {u, w}
                    
                    # Extend clique
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x not in clique:
                                if all((min(x, y), max(x, y)) in rem_edges_set for y in clique):
                                    clique.add(x)
                                    changed = True
                                    break
                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(clique, 2):
                            rem_edges_set.discard((min(a, b), max(a, b)))
                    else:
                        cliques.append([u, w])
                        rem_edges_set.discard(e)
            
            clean_cliques = [[int(x) for x in c] for c in cliques]
            count = len(clean_cliques)
            
            if count < best_count:
                best_count = count
                best_cliques = clean_cliques
        
        if best_cliques is None:
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
