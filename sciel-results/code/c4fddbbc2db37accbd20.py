"""
Clique partition via Simulated Annealing on Edge-Clique Assignments.
Structural change: Instead of a monotonic greedy merge strategy, we use 
Simulated Annealing which allows non-improving moves (increasing the number 
of cliques temporarily) to escape local minima. This enables "splitting" 
cliques to find better global configurations that a greedy merge cannot reach.
"""
import time
import random
import math
import sys

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

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    if not cliques:
        return len(edge_list) == 0
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _solve_one_graph(adj, edge_list, time_budget=50.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []
    
    m = len(edge_list)
    
    # Precompute compatibility for edges to be in the same clique.
    # Two edges e1=(a,b) and e2=(c,d) are compatible if {a,b,c,d} is a clique.
    # We store this as a boolean matrix or set of sets for O(1) lookup.
    # For m~1100, a list of sets is fine.
    compat = [set() for _ in range(m)]
    for i in range(m):
        a, b = edge_list[i]
        for j in range(i + 1, m):
            c, d = edge_list[j]
            # Check if all pairs in {a,b,c,d} are edges in G
            # Since (a,b) and (c,d) are edges, we just need to check cross-edges
            # and potential duplicates if they share vertices.
            # Vertices involved:
            verts = {a, b, c, d}
            ok = True
            for x in verts:
                for y in verts:
                    if x < y and not adj[x][y]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                compat[i].add(j)
                compat[j].add(i)
    
    # Initial Solution: Greedy Coloring (Welsh-Powell style)
    # Order edges by degree in conflict graph (inverse of compat)
    # Actually, just random order or degree order is fine for initialization.
    order = list(range(m))
    random.Random(123).shuffle(order)
    
    # color_of_edge[i] = index of the clique
    # cliques is a list of sets of edge indices
    color_of_edge = [-1] * m
    cliques = []
    
    for eidx in order:
        placed = False
        # Try to put in existing clique
        for ci, c_edges in enumerate(cliques):
            # Check if eidx is compatible with all edges in c_edges
            can_add = True
            for other_e in c_edges:
                if other_e not in compat[eidx]:
                    can_add = False
                    break
            if can_add:
                color_of_edge[eidx] = ci
                c_edges.add(eidx)
                placed = True
                break
        if not placed:
            color_of_edge[eidx] = len(cliques)
            cliques.append({eidx})
            
    def _get_clique_vertices(c_edges):
        verts = set()
        for eidx in c_edges:
            u, v = edge_list[eidx]
            verts.add(u)
            verts.add(v)
        return sorted(verts)
        
    current_cliques = [_get_clique_vertices(c) for c in cliques]
    if not _verify_partition(current_cliques, edge_list, adj):
        return [[u, v] for u, v in edge_list]
    
    best_cliques = current_cliques
    best_count = len(current_cliques)
    
    # Simulated Annealing
    # State: cliques (list of sets of edge indices)
    # Move: Pick a random edge, move it to a random other clique (if compatible) 
    #       or to a new clique.
    # Energy: len(cliques)
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    current_cliques_sets = [set(c) for c in cliques]
    current_count = len(current_cliques_sets)
    
    # Precompute for each edge, the list of cliques it is compatible with?
    # This is dynamic, so we check on the fly.
    
    while time.time() - start_time < time_budget and T > min_T:
        if _remaining() < 5:
            break
            
        # Pick a random edge
        eidx = random.randint(0, m - 1)
        current_clique_idx = color_of_edge[eidx]
        
        # Pick a destination
        # 1. Move to existing clique
        # 2. Move to new clique
        
        # With probability 0.9, try to move to existing
        # With probability 0.1, try to move to new (split)
        
        if random.random() < 0.9:
            # Pick a random target clique different from current
            if current_count <= 1:
                T *= cooling_rate
                continue
                
            target_idx = random.randint(0, current_count - 1)
            if target_idx == current_clique_idx:
                continue
                
            target_set = current_cliques_sets[target_idx]
            
            # Check if eidx is compatible with all edges in target_set
            can_add = True
            for other_e in target_set:
                if other_e not in compat[eidx]:
                    can_add = False
                    break
            
            if can_add:
                # Perform move
                current_cliques_sets[current_clique_idx].discard(eidx)
                if not current_cliques_sets[current_clique_idx]:
                    # Remove empty clique
                    # Need to update color_of_edge for all edges in other cliques?
                    # No, we just remove the set. But indices shift.
                    # To avoid index shifting complexity, we keep empty sets and ignore them in count?
                    # Or we compact. Compacting is O(N). Let's just mark as empty and filter later.
                    # Actually, let's just remove and re-index is messy.
                    # Let's keep the list of sets, but if one becomes empty, we can leave it empty.
                    # The "count" is number of non-empty sets.
                    pass
                
                current_cliques_sets[target_idx].add(eidx)
                color_of_edge[eidx] = target_idx
                
                new_count = sum(1 for s in current_cliques_sets if s)
                delta_E = new_count - current_count
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_cliques = [_get_clique_vertices(s) for s in current_cliques_sets if s]
                else:
                    # Revert
                    current_cliques_sets[target_idx].discard(eidx)
                    current_cliques_sets[current_clique_idx].add(eidx)
                    color_of_edge[eidx] = current_clique_idx
        else:
            # Move to new clique (split)
            current_cliques_sets[current_clique_idx].discard(eidx)
            if not current_cliques_sets[current_clique_idx]:
                pass # It becomes empty
            
            new_idx = len(current_cliques_sets)
            current_cliques_sets.append({eidx})
            color_of_edge[eidx] = new_idx
            
            new_count = sum(1 for s in current_cliques_sets if s)
            delta_E = new_count - current_count # Usually +1
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_count = new_count
                # No improvement in count expected here, but might enable future merges
            else:
                # Revert
                current_cliques_sets.pop()
                current_cliques_sets[current_clique_idx].add(eidx)
                color_of_edge[eidx] = current_clique_idx
                
        T *= cooling_rate
        
        # Occasionally, try to merge two cliques if possible?
        # The "move to existing" effectively does this if we pick the right target.
        
    # Final cleanup: convert sets to vertex lists
    final_cliques = []
    for s in current_cliques_sets:
        if s:
            final_cliques.append(_get_clique_vertices(s))
            
    if not _verify_partition(final_cliques, edge_list, adj):
        return best_cliques if best_cliques else [[u, v] for u, v in edge_list]
        
    return final_cliques

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        time_budget = min(55.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
