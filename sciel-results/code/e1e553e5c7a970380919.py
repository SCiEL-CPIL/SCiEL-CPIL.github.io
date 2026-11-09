import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Enumerate all maximal cliques using Bron-Kerbosch
    # We will use a recursive implementation with pivot selection for speed.
    maximal_cliques = []
    
    def bronk(R, P, X):
        if not _deadline_ok():
            return
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(list(R))
            return
        
        # Pivot selection: choose u in P U X with max |P intersect N(u)|
        u = None
        max_size = -1
        for v in P:
            if u is None:
                u = v
                max_size = len(P & adj[v])
            else:
                size = len(P & adj[v])
                if size > max_size:
                    max_size = size
                    u = v
        
        if u is None:
            # P is empty, handled above
            return
            
        # Candidates are P \ N(u)
        candidates = list(P - adj[u])
        
        for v in candidates:
            # Recurse
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    bronk(set(), set(range(n)), set())
    
    # If no cliques found (should not happen if edges exist), fallback
    if not maximal_cliques:
        return [[u, v] for u, v in edges]
    
    # 2. Prepare data for Beam Search
    # Map each edge to the list of indices of maximal cliques that contain it
    edge_to_cliques = {}
    clique_edges = [] # list of frozensets of edges for each clique
    
    for i, clique in enumerate(maximal_cliques):
        c_edges = frozenset((u, v) if u < v else (v, u) for u, v in combinations(clique, 2))
        clique_edges.append(c_edges)
        for e in c_edges:
            if e not in edge_to_cliques:
                edge_to_cliques[e] = []
            edge_to_cliques[e].append(i)
            
    all_edges_set = frozenset(edges)
    
    # 3. Beam Search
    # State: (remaining_edges_frozenset, clique_count, chosen_clique_indices_tuple)
    # We store the chosen indices to reconstruct the solution.
    # To save memory/time, we might just store the partition if it's small, 
    # but storing indices is safer.
    
    beam_width = 20
    best_solution = None
    best_count = float('inf')
    
    # Initial state
    # We need to pick an edge.
    # Heuristic: Pick the edge with the fewest candidate cliques.
    
    def get_most_constrained_edge(remaining_edges):
        min_count = float('inf')
        best_edge = None
        for e in remaining_edges:
            # Count how many cliques in our pool contain this edge AND are "compatible" 
            # (i.e., all their edges are in remaining_edges)
            # For speed, we just use the global count of cliques containing e.
            # A more refined heuristic would check compatibility, but that's O(C).
            count = len(edge_to_cliques.get(e, []))
            if count < min_count:
                min_count = count
                best_edge = e
        return best_edge

    # Beam: List of (remaining_edges, count, path)
    # path is a tuple of clique indices
    beam = [(all_edges_set, 0, ())]
    
    while beam and _deadline_ok():
        next_beam = {}
        
        for state in beam:
            remaining, count, path = state
            
            if not remaining:
                # Found a complete solution
                if count < best_count:
                    best_count = count
                    best_solution = path
                continue
            
            # If we have already found a solution, and this path is already worse 
            # (count >= best_count), we can prune it? 
            # Not necessarily, because we might find a better one later? 
            # No, count only increases. If count >= best_count, this branch cannot improve.
            if count >= best_count:
                continue
                
            # Pick the most constrained edge
            edge = get_most_constrained_edge(remaining)
            if edge is None:
                continue
                
            # Get candidates
            candidates = edge_to_cliques.get(edge, [])
            
            # Sort candidates by size (larger cliques first) to find good solutions early
            # This helps prune the beam if we find a good solution quickly
            candidates_sorted = sorted(candidates, key=lambda idx: -len(maximal_cliques[idx]))
            
            for idx in candidates_sorted:
                c_edges = clique_edges[idx]
                # Check if this clique is compatible (all its edges are in remaining)
                # c_edges is a frozenset
                if c_edges.issubset(remaining):
                    new_remaining = remaining - c_edges
                    new_count = count + 1
                    new_path = path + (idx,)
                    
                    # Add to next_beam
                    # We use new_remaining as key. If we have multiple paths leading to the same remaining set,
                    # we only keep the one with the lowest count.
                    # Actually, the path matters for reconstruction, but if remaining is same and count is same, 
                    # either path is fine? 
                    # Wait, if remaining is same, the future choices are the same. 
                    # So we can deduplicate by remaining_edges.
                    
                    if new_remaining in next_beam:
                        if new_count < next_beam[new_remaining][0]:
                            next_beam[new_remaining] = (new_count, new_path)
                    else:
                        next_beam[new_remaining] = (new_count, new_path)
                        
        # Prune next_beam to keep only the best 'beam_width' states
        if not next_beam:
            break
            
        # Sort by count
        sorted_states = sorted(next_beam.items(), key=lambda x: x[1][0])
        
        # Keep top K
        beam = []
        for rem, (cnt, pth) in sorted_states[:beam_width]:
            beam.append((rem, cnt, pth))
            
        # Optimization: If we found a solution, we can increase beam width or just continue
        # If best_count is small, we might not need wide beam.
        
    # Fallback if beam search failed or timed out
    if best_solution is None:
        # Fallback to greedy peeling (simple)
        current_edges = set(edges)
        partition_indices = []
        while current_edges:
            if not _deadline_ok():
                break
            # Pick first edge
            e = next(iter(current_edges))
            # Find a clique containing e
            found = False
            for idx, c_edges in enumerate(clique_edges):
                if e in c_edges and c_edges.issubset(current_edges):
                    partition_indices.append(idx)
                    current_edges -= c_edges
                    found = True
                    break
            if not found:
                # Should not happen if we have all maximal cliques
                # But if the graph is sparse, maybe?
                # Just add the edge as a clique of size 2
                # We need to map this back to a vertex list
                # This fallback is messy. Let's just use the greedy vertex expansion from before.
                # For simplicity in fallback, return trivial partition
                return [[u, v] for u, v in edges]
        
        best_solution = partition_indices
        best_count = len(best_solution)
        
    # Reconstruct solution from indices
    result = []
    for idx in best_solution:
        result.append(maximal_cliques[idx])
        
    return result

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
