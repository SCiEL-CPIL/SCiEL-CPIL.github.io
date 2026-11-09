"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n = 16
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Build Q3 adjacency lists
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def check_q3_free(adj):
        """Check if graph with adjacency matrix adj is Q3-free using backtracking."""
        # Try all 8-vertex subsets
        for subset in itertools.combinations(range(n), 8):
            subset_set = set(subset)
            # Backtracking: assign Q3 vertices to subset vertices
            assignment = [-1] * 8
            used = set()
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    return True
                for sv in subset:
                    if sv in used:
                        continue
                    # Check compatibility with already assigned neighbors
                    compatible = True
                    for prev_q3 in range(q3_idx):
                        if prev_q3 in q3_adj[q3_idx]:
                            prev_sv = assignment[prev_q3]
                            if not adj[sv][prev_sv]:
                                compatible = False
                                break
                    if compatible:
                        used.add(sv)
                        assignment[q3_idx] = sv
                        if backtrack(q3_idx + 1):
                            return True
                        used.remove(sv)
                        assignment[q3_idx] = -1
                return False
            
            if backtrack(0):
                return False  # Found a Q3 subgraph
        
        return True  # No Q3 subgraph found
    
    def check_q3_free_fast(adj_sets, new_edge):
        """Fast check: only verify if adding new_edge creates a Q3 using structural properties."""
        u, v = new_edge
        
        # Get neighbors of u and v
        nu = list(adj_sets[u])
        nu.remove(v)
        nv = list(adj_sets[v])
        nv.remove(u)
        
        # We assume the edge (u,v) maps to (0,1) in Q3.
        # Q3 structure relative to edge (0,1):
        # 0 is connected to 1, 2, 4.
        # 1 is connected to 0, 3, 5.
        # 2 connected to 0, 6.
        # 4 connected to 0, 6.
        # 3 connected to 1, 7.
        # 5 connected to 1, 7.
        # 6 connected to 2, 4, 7.
        # 7 connected to 3, 5, 6.
        
        # So we need:
        # x, y in N(u) (images of 2, 4)
        # z, w in N(v) (images of 3, 5)
        # p in N(x) \cap N(y) (image of 6)
        # q in N(z) \cap N(w) (image of 7)
        # p-q is an edge
        # All distinct: u, v, x, y, z, w, p, q
        
        for i in range(len(nu)):
            x = nu[i]
            for j in range(i+1, len(nu)):
                y = nu[j]
                # x, y distinct from u, v by construction of nu
                # x != y by loop
                for k in range(len(nv)):
                    z = nv[k]
                    if z == x or z == y:
                        continue
                    for l in range(k+1, len(nv)):
                        w = nv[l]
                        if w == x or w == y or w == z:
                            continue
                        
                        # Find candidates for p: neighbors of x and y
                        # p must be distinct from u, v, x, y, z, w
                        # p is in N(x) \cap N(y)
                        common_xy = adj_sets[x] & adj_sets[y]
                        forbidden = {u, v, x, y, z, w}
                        p_candidates = common_xy - forbidden
                        if not p_candidates:
                            continue
                        
                        common_zw = adj_sets[z] & adj_sets[w]
                        # q must be distinct from u, v, x, y, z, w and p
                        # We can't subtract p yet because p is not fixed
                        # But we can check for each p
                        for p in p_candidates:
                            forbidden_q = {u, v, x, y, z, w, p}
                            q_candidates = common_zw - forbidden_q
                            for q in q_candidates:
                                if q in adj_sets[p]:
                                    return False
                            
        return True
    
    # Start with K_{3,13}: vertices 0,1,2 in part A, vertices 3-15 in part B
    adj = [[0]*n for _ in range(n)]
    adj_sets = [set() for _ in range(n)]
    edges = []
    
    # Add all edges between part A (0,1,2) and part B (3-15)
    for a in range(3):
        for b in range(3, n):
            adj[a][b] = 1
            adj[b][a] = 1
            adj_sets[a].add(b)
            adj_sets[b].add(a)
            edges.append([a, b])
    
    # Now greedily add all other possible edges
    all_possible_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                all_possible_edges.append((i, j))
    
    # Sort edges: prioritize edges that are "easier" to add
    # Heuristic: edges within smaller partitions first (less likely to create Q3)
    # Part A has 3 vertices, Part B has 13 vertices
    # Edges within A: (0,1), (0,2), (1,2) - very safe
    # Edges within B: many, riskier
    # Cross edges already added
    
    # Priority: within-A edges first, then cross (already done), then within-B
    def edge_priority(edge):
        i, j = edge
        in_a = lambda x: x < 3
        if in_a(i) and in_a(j):
            return 0  # Highest priority
        elif not in_a(i) and not in_a(j):
            return 1  # Lower priority
        else:
            return 2  # Shouldn't happen (cross edges already added)
    
    all_possible_edges.sort(key=edge_priority)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    for edge in all_possible_edges:
        if time.time() > deadline - 5:
            break
        
        u, v = edge
        # Try adding this edge
        adj[u][v] = 1
        adj[v][u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edges.append([u, v])
        
        # Check if Q3-free
        if check_q3_free_fast(adj_sets, (u, v)):
            best_edges = list(edges)
            best_count = len(edges)
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            edges.pop()
    
    return best_edges
# EVOLVE-BLOCK-END
