"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    n = 18
    
    # Adjacency as set of frozensets for fast lookup
    # Start with complete tripartite K6,6,6
    parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
    adj = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj.add(frozenset([a, b]))
    
    def is_k4_free(adj_set):
        # Check no K4: for every 4-clique candidate
        # A K4 exists iff there are 4 vertices all pairwise connected
        # Efficient: for each edge (u,v), check if any pair (w,x) from common neighbors forms K4
        # Simpler: just check all 4-subsets (C(18,4) = 3060)
        verts = list(range(n))
        for quad in combinations(verts, 4):
            a, b, c, d = quad
            if (frozenset([a,b]) in adj_set and frozenset([a,c]) in adj_set and 
                frozenset([a,d]) in adj_set and frozenset([b,c]) in adj_set and
                frozenset([b,d]) in adj_set and frozenset([c,d]) in adj_set):
                return False
        return True
    
    def max_triangle_free_induced(adj_set):
        # Find largest subset of vertices inducing a triangle-free graph
        # Use branch and bound
        verts = list(range(n))
        best = 0
        
        # Precompute: for each vertex, its neighbors
        neighbors = [set() for _ in range(n)]
        for edge in adj_set:
            u, v = tuple(edge)
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        # A set S is triangle-free iff no three vertices in S are all mutually adjacent
        # i.e., no triangle is fully contained in S
        
        # Branch and bound: build up a triangle-free set
        # State: (current_set, candidates)
        # candidates are vertices that can be added without creating a triangle
        
        def can_add_triangle_free(current_set, v, adj_set, neighbors):
            # Adding v to current_set creates a triangle if there exist u,w in current_set 
            # such that (u,v) in adj and (w,v) in adj and (u,w) in adj
            # i.e., v's neighbors in current_set must be an independent set
            vs_in_set = [x for x in current_set if x in neighbors[v]]
            # Check if vs_in_set is independent
            for i in range(len(vs_in_set)):
                for j in range(i+1, len(vs_in_set)):
                    if frozenset([vs_in_set[i], vs_in_set[j]]) in adj_set:
                        return False
            return True
        
        # Recursive search with pruning
        # We want to find the max size triangle-free induced subgraph
        # Use the fact that if remaining candidates + current size <= best, prune
        
        def search(current_set, candidates, depth):
            nonlocal best
            if time.time() > deadline - 5:
                return
            if len(candidates) + len(current_set) <= best:
                return
            if not candidates:
                if len(current_set) > best:
                    best = len(current_set)
                return
            
            # Pick a candidate
            v = candidates[0]
            remaining = candidates[1:]
            
            # Option 1: include v
            if can_add_triangle_free(current_set, v, adj_set, neighbors):
                new_current = current_set | {v}
                # Filter remaining: only those compatible with new_current
                new_cands = [x for x in remaining if can_add_triangle_free(new_current, x, adj_set, neighbors)]
                search(new_current, new_cands, depth + 1)
            
            # Option 2: exclude v
            search(current_set, remaining, depth + 1)
        
        search(set(), verts, 0)
        return best
    
    # Verify initial graph
    if not is_k4_free(adj):
        # Fallback
        parts_list = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts_list[i] for b in parts_list[j]]
    
    current_best = max_triangle_free_induced(adj)
    
    # Local search: try edge additions and removals
    all_possible_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_possible_edges.add(frozenset([u, v]))
    
    random.seed(42)
    
    while time.time() < deadline - 10:
        # Try random edge modifications
        # Add a random missing edge
        missing = all_possible_edges - adj
        if missing:
            candidate_edge = random.choice(list(missing))
            new_adj = adj | {candidate_edge}
            if is_k4_free(new_adj):
                score = max_triangle_free_induced(new_adj)
                if score < current_best:
                    adj = new_adj
                    current_best = score
                    continue
        # Remove a random existing edge
        if adj:
            candidate_edge = random.choice(list(adj))
            new_adj = adj - {candidate_edge}
            # Removing edges always preserves K4-free
            score = max_triangle_free_induced(new_adj)
            if score < current_best:
                adj = new_adj
                current_best = score
    
    # Convert adj set back to edge list
    result = []
    for edge in adj:
        u, v = tuple(edge)
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
