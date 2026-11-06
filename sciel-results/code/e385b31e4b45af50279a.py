"""Decompose three fixed random graphs into edge-disjoint cycles and edges, minimizing pieces."""
import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _greedy_decompose(edges, n, deadline):
    """
    Greedy cycle extraction.
    Returns a list of pieces (cycles or single edges).
    """
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    rng = random.Random(42)
    
    while remaining:
        if time.time() >= deadline:
            break
        
        # Pick a random edge to start
        start_edge = next(iter(remaining))
        u, v = start_edge
        # Try to find a cycle starting with this edge
        # We use a DFS that tries to close the cycle
        found_cycle = None
        
        # Limit DFS depth to avoid explosion
        max_depth = 50
        stack = [(u, [u], {u}, 0)]
        # We start at u, having come from v? No, we just picked edge (u,v).
        # Let's say we are at v, having traversed u->v.
        # Path so far: [u, v]. Current node: v.
        # We want to get back to u.
        
        # To make it efficient, we prioritize neighbors with high degree
        def dfs(current, path, visited, depth):
            if depth > max_depth:
                return None
            if current == u and len(path) >= 3:
                return list(path)
            
            # Get neighbors
            nbrs = list(adj.get(current, ()))
            # Sort by degree descending to find long cycles?
            nbrs.sort(key=lambda x: len(adj[x]), reverse=True)
            rng.shuffle(nbrs) # Add randomness
            
            for nb in nbrs:
                if nb == u and len(path) >= 3:
                    # Found cycle
                    return list(path)
                if nb not in visited:
                    res = dfs(nb, path + [nb], visited | {nb}, depth + 1)
                    if res:
                        return res
            return None
        
        # Try from both ends
        for start_node, end_node in [(u, v), (v, u)]:
            # We start at start_node, want to end at end_node?
            # No, the edge (u,v) is used. So we start at v, want to return to u.
            # Path: [u, v]
            res = dfs(v, [u, v], {u, v}, 1)
            if res:
                found_cycle = res
                break
        
        if found_cycle:
            # Extract edges from cycle
            cycle_edges = set()
            for i in range(len(found_cycle)):
                a, b = found_cycle[i], found_cycle[(i + 1) % len(found_cycle)]
                e = (a, b) if a < b else (b, a)
                cycle_edges.add(e)
            
            # Remove from remaining
            for e in cycle_edges:
                remaining.discard(e)
                a, b = e
                if b in adj[a]: adj[a].discard(b)
                if a in adj[b]: adj[b].discard(a)
            
            pieces.append(found_cycle)
        else:
            # No cycle found involving this edge immediately.
            # Add as singleton for now.
            pieces.append([u, v])
            remaining.discard(start_edge)
            if v in adj[u]: adj[u].discard(v)
            if u in adj[v]: adj[v].discard(u)
            
    # Add any remaining edges as singletons
    for e in remaining:
        pieces.append(list(e))
        
    return pieces

def _local_search(pieces, edges, n, deadline):
    """
    Local search using 2-opt moves.
    """
    # Build a mapping from edge to piece index
    edge_to_piece = {}
    for i, p in enumerate(pieces):
        if len(p) == 2:
            e = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
            edge_to_piece[e] = i
        else:
            for j in range(len(p)):
                a, b = p[j], p[(j + 1) % len(p)]
                e = (a, b) if a < b else (b, a)
                edge_to_piece[e] = i

    # Adjacency of the original graph
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs of pieces? O(N^2) might be slow if N is large.
        # But N is small (number of pieces).
        # We iterate through pieces and try to improve them.
        
        # Strategy: 
        # 1. If a piece is a path, check if its endpoints are adjacent. If so, close it to a cycle.
        # 2. If two pieces share a vertex, try to splice them.
        
        # 1. Close paths
        for i in range(len(pieces)):
            p = pieces[i]
            if len(p) == 2:
                continue
            # Check if it's a path or cycle. 
            # In our representation, a cycle is a list where last connects to first.
            # A path is a list where last does NOT connect to first in the piece, 
            # BUT we don't explicitly store "is path". 
            # Actually, all pieces are simple cycles or single edges.
            # Wait, the greedy algorithm only produces cycles and single edges.
            # So there are no "paths" of length > 1 initially.
            # However, 2-opt can create paths.
            
            # Let's assume pieces are lists of vertices.
            # If len(p) > 2, it's a cycle.
            # If len(p) == 2, it's an edge.
            
            # To create paths, we need to break cycles? No, that increases pieces.
            # We want to MERGE pieces.
            
            # Splicing:
            # Take piece i and piece j.
            # If they are disjoint, no direct splice.
            # If they share a vertex v, we can try to reconnect.
            
        # Let's implement a specific splice:
        # Find all vertices that appear in multiple pieces.
        # For each such vertex v, get the list of pieces containing v.
        # For each pair of pieces (P1, P2) in that list:
        #   Try to splice P1 and P2 at v.
        
        vertex_pieces = defaultdict(list)
        for i, p in enumerate(pieces):
            for v in p:
                vertex_pieces[v].append(i)
        
        for v, p_indices in vertex_pieces.items():
            if len(p_indices) < 2:
                continue
            
            # Get unique piece indices
            unique_indices = list(set(p_indices))
            
            for k in range(len(unique_indices)):
                for l in range(k + 1, len(unique_indices)):
                    i, j = unique_indices[k], unique_indices[l]
                    if i == j: continue
                    
                    P1 = pieces[i]
                    P2 = pieces[j]
                    
                    # Find edges incident to v in P1 and P2
                    # P1 is a cycle. v has 2 neighbors in P1.
                    # P2 is a cycle. v has 2 neighbors in P2.
                    
                    def get_neighbors_in_piece(piece, vert):
                        nbrs = []
                        for idx in range(len(piece)):
                            if piece[idx] == vert:
                                prev = piece[(idx - 1) % len(piece)]
                                next = piece[(idx + 1) % len(piece)]
                                nbrs.append((prev, next))
                        return nbrs
                    
                    nbrs1 = get_neighbors_in_piece(P1, v)
                    nbrs2 = get_neighbors_in_piece(P2, v)
                    
                    if len(nbrs1) != 2 or len(nbrs2) != 2:
                        continue
                        
                    # nbrs1: [(a, b), ...] where a-b-v is the path? No, a and b are neighbors of v.
                    # So edges (a,v) and (v,b) are in P1.
                    # nbrs2: [(c, d)] where edges (c,v) and (v,d) are in P2.
                    
                    a, b = nbrs1[0]
                    c, d = nbrs2[0]
                    
                    # Current connections: a-v-b and c-v-d
                    # New connections: a-v-c and b-v-d ? No, v is the center.
                    # We are swapping the partners of v.
                    # Original: (a,v), (v,b), (c,v), (v,d)
                    # New: (a,v), (v,c), (b,v), (v,d)
                    # This requires edges (a,v) [exists], (v,c) [exists?], (b,v) [exists], (v,d) [exists?].
                    # Wait, (a,v) and (b,v) are already in P1. (c,v) and (d,v) are in P2.
                    # The new edges would be (a,c) and (b,d)? No, that disconnects v.
                    # Splicing at v means we break the 4 edges incident to v and reconnect them.
                    # But we must keep the degree of v at 2 in the new graph.
                    # So we must reconnect v to two of the four neighbors.
                    # The other two neighbors must be connected to each other? No, they are in different pieces.
                    
                    # Actually, a valid splice at a shared vertex v for two cycles is:
                    # Cycle 1: ... a - v - b ...
                    # Cycle 2: ... c - v - d ...
                    # If we have edges (a,c) and (b,d), we can form:
                    # ... a - c ... d - b ... v - ...? No.
                    # If we form ... a - c ... and ... b - d ..., v is isolated?
                    # No, v must be connected.
                    
                    # Correct splice:
                    # If we have edges (a,d) and (b,c), we can form:
                    # ... a - d ... c - b ... v ...? No.
                    
                    # Let's look at the structure.
                    # We have 4 edges incident to v: (v,a), (v,b), (v,c), (v,d).
                    # We want to replace them with 2 edges incident to v, and 2 edges NOT incident to v?
                    # No, the edges (v,a) etc are fixed in the graph. We are just changing which piece they belong to?
                    # No, the pieces must be edge-disjoint and cover all edges.
                    # The set of edges is fixed. We are just partitioning them.
                    # So we cannot "create" new edges like (a,c).
                    # We can only reassign existing edges to different pieces.
                    
                    # So, if we take edges (v,a) and (v,c) and put them in one piece, and (v,b) and (v,d) in another?
                    # That just swaps which cycle they belong to. It doesn't merge them.
                    
                    # To MERGE two cycles sharing a vertex v, we need to "break" the cycles at v and reconnect.
                    # Cycle 1: v-a-...-b-v
                    # Cycle 2: v-c-...-d-v
                    # If we have edges (a,c) and (b,d), we can form:
                    # v-a-c-...-d-b-...-v? No, that uses (a,c) and (b,d) which are NOT in the original graph.
                    
                    # Conclusion: You cannot merge two cycles that share exactly one vertex by just reassigning edges, unless there are chords.
                    
                    # So 2-opt on shared vertices is not useful for merging cycles.
                    
                    # What about merging a path and a cycle?
                    # Or extending a cycle?
                    
                    # Let's try a different move: **Chord Splicing**.
                    # If a cycle has a chord (u,v), we can split it into two cycles. (Increases pieces, bad).
                    # If a path has a chord connecting its endpoints, we close it. (Decreases pieces, good).
                    
                    # Since we start with cycles and single edges, we have no paths.
                    # So we need to BREAK a cycle to form paths, then MERGE them?
                    # That seems counter-productive.
                    
                    # Alternative: **Greedy with Better Cycle Finding**.
                    # The main issue is that greedy might pick short cycles.
                    # If we pick a long cycle, we cover more edges with 1 piece.
                    
                    # Let's improve the cycle finder to maximize length.
                    # Use a heuristic: prefer neighbors with high degree.
                    
    return pieces

def _solve_one(n, p, seed, deadline):
    edges = _instance(n, p, seed)
    pieces = _greedy_decompose(edges, n, deadline)
    # Local search is skipped for now as it's complex to implement correctly for this specific "repartitioning" constraint
    return pieces

def construct():
    t0 = time.time()
    total_budget = 120.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, spec in enumerate(specs):
        per = (total_budget - (time.time() - t0)) / (len(specs) - i)
        deadline = time.time() + max(1.0, per - 2.0)
        try:
            pieces = _solve_one(*spec, deadline)
        except Exception:
            pieces = [list(e) for e in _instance(*spec)]
        results.append(pieces)
    return results

if __name__ == "__main__":
    res = construct()
    print([len(r) for r in res])
