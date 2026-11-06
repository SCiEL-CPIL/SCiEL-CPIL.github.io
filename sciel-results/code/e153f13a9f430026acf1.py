"""
Clique partition via simulated annealing on edge-clique assignments.
Strategy:
- Start with each edge as its own clique (fine partition).
- Use simulated annealing with moves: merge two cliques if union is a clique,
  or move an edge from one clique to another if the target remains a clique.
- Objective: minimize number of cliques.
- Deadline guard to ensure we return before 150s.
"""
import time
import random
from collections import defaultdict

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _is_clique(verts, adj):
    verts_list = list(verts)
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if not adj[verts_list[i]][verts_list[j]]:
                return False
    return True

def _clique_edges(clique, adj):
    verts = list(clique)
    edges = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if adj[verts[i]][verts[j]]:
                edges.add((min(verts[i], verts[j]), max(verts[i], verts[j])))
    return edges

def _build_edge_to_clique(cliques, adj):
    edge_map = {}
    for ci, clique in enumerate(cliques):
        for e in _clique_edges(clique, adj):
            edge_map[e] = ci
    return edge_map

def _solve_one(adj, edge_list, time_budget):
    n = len(edge_list[0][0]) if edge_list else 0
    if not edge_list:
        return []
    
    start_time = time.time()
    max_time = min(time_budget, _remaining() - 5)
    if max_time <= 1:
        return [[u, v] for u, v in edge_list]
    
    # Initial: each edge is its own clique
    cliques = [set([u, v]) for u, v in edge_list]
    rng = random.Random(42)
    
    # Precompute for speed: adjacency as sets
    adj_sets = [set() for _ in range(n)]
    for u, v in edge_list:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    def is_clique_fast(verts):
        verts_list = list(verts)
        for i in range(len(verts_list)):
            vi = verts_list[i]
            for j in range(i + 1, len(verts_list)):
                if verts_list[j] not in adj_sets[vi]:
                    return False
        return True
    
    def get_clique_edges_fast(clique):
        verts = list(clique)
        edges = set()
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                a, b = verts[i], verts[j]
                if b in adj_sets[a]:
                    edges.add((min(a, b), max(a, b)))
        return edges
    
    # Objective: number of cliques
    def objective(cliques):
        return len(cliques)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    best_cliques = [set(c) for c in cliques]
    best_obj = objective(cliques)
    
    while time.time() - start_time < max_time and T > T_min:
        if _remaining() < 10:
            break
        
        # Random move
        move_type = rng.random()
        
        if move_type < 0.6 and len(cliques) >= 2:
            # Try to merge two cliques
            i, j = rng.sample(range(len(cliques)), 2)
            union = cliques[i] | cliques[j]
            if is_clique_fast(union):
                # Accept merge
                cliques[i] = union
                cliques.pop(j)
                new_obj = objective(cliques)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_cliques = [set(c) for c in cliques]
            # Don't un-merge; just skip if not a clique
        
        elif move_type < 0.9 and len(cliques) >= 2:
            # Move an edge from one clique to another
            # Pick a random clique with >2 vertices
            candidates = [i for i, c in enumerate(cliques) if len(c) > 2]
            if candidates:
                src = rng.choice(candidates)
                dst = rng.choice([i for i in range(len(cliques)) if i != src])
                # Pick a random edge in src
                src_edges = list(get_clique_edges_fast(cliques[src]))
                if src_edges:
                    edge = rng.choice(src_edges)
                    u, v = edge
                    # Try to add u,v to dst
                    new_dst = cliques[dst] | {u, v}
                    if is_clique_fast(new_dst):
                        # Remove edge from src: remove u,v if not needed
                        src_verts = set(cliques[src])
                        other_edges = get_clique_edges_fast(cliques[src]) - {edge}
                        u_needed = any(u in e for e in other_edges)
                        v_needed = any(v in e for e in other_edges)
                        if not u_needed:
                            src_verts.discard(u)
                        if not v_needed:
                            src_verts.discard(v)
                        if len(src_verts) >= 2:
                            cliques[src] = src_verts
                            cliques[dst] = new_dst
                            new_obj = objective(cliques)
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_cliques = [set(c) for c in cliques]
        
        else:
            # Split a clique into two smaller cliques (to escape local minima)
            candidates = [i for i, c in enumerate(cliques) if len(c) > 3]
            if candidates:
                i = rng.choice(candidates)
                verts = list(cliques[i])
                if len(verts) >= 4:
                    # Split into two groups
                    mid = len(verts) // 2
                    group1 = set(verts[:mid])
                    group2 = set(verts[mid:])
                    if is_clique_fast(group1) and is_clique_fast(group2):
                        cliques[i] = group1
                        cliques.append(group2)
        
        T *= cooling
    
    # Final: try greedy merges on best
    cliques = [set(c) for c in best_cliques]
    improved = True
    while improved and _remaining() > 5:
        improved = False
        n_c = len(cliques)
        for i in range(n_c):
            for j in range(i + 1, n_c):
                union = cliques[i] | cliques[j]
                if is_clique_fast(union):
                    cliques[i] = union
                    cliques.pop(j)
                    improved = True
                    break
            if improved:
                break
    
    return [sorted(c) for c in cliques]

def _verify_partition(cliques, edge_list, adj):
    edge_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    
    for clique in cliques:
        verts = list(clique)
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                if not adj[verts[a]][verts[b]]:
                    return False
                e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 15:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        time_budget = min(60.0, _remaining() - 10)
        cliques = _solve_one(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
