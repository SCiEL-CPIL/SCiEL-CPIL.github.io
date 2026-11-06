"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_c5(adj, n):
    """Check if graph has a 5-cycle using adjacency matrix."""
    # A 5-cycle exists if there are vertices v1,v2,v3,v4,v5 with edges
    # v1-v2, v2-v3, v3-v4, v4-v5, v5-v1 and no other constraints (simple cycle)
    # Use: for each pair of non-adjacent vertices, check if they share two common neighbors
    # that are also non-adjacent to each other (forming a path of length 4)
    # Actually, simpler: a C5 exists iff there exist v1,v2,v3,v4,v5 distinct with
    # adj[v1][v2] & adj[v2][v3] & adj[v3][v4] & adj[v4][v5] & adj[v5][v1]
    # and NOT (adj[v1][v3] & adj[v2][v4]) etc. - but for subgraph C5, we just need
    # the 5 edges to exist.
    # For n<=35, we can enumerate. But let's use a smarter approach.
    # C5-free means no induced or non-induced 5-cycle.
    # A 5-cycle is a set of 5 vertices with exactly the cycle edges (at least the 5 cycle edges).
    # For subgraph containment: 5 vertices with edges forming a cycle of length 5.
    # We can check: does there exist a set of 5 vertices where the induced subgraph
    # contains a 5-cycle?
    # Faster: for each 5-tuple (v1,v2,v3,v4,v5) in order, check adj[v1][v2], adj[v2][v3],
    # adj[v3][v4], adj[v4][v5], adj[v5][v1].
    # But that's O(n^5) which for n=35 is ~52M - too slow per check.
    # Better approach: for each pair (u,v) non-adjacent, count common neighbors.
    # If u and v have common neighbors a,b where a-b is an edge, and a,c where c is another...
    # Actually, let me think differently.
    # A C5 can be detected by: for each vertex v, look at its neighbors.
    # If two neighbors of v are connected to a common vertex w (w != v), and w connects to
    # another neighbor of v... this is getting complex.
    # For n <= 35, let's use: for each 5-subset, check if it contains a C5.
    # But that's C(35,5) = 324632 subsets, each needing a check.
    # Alternative: use the fact that C5-free graphs have structural properties.
    # Let me just do the direct check but optimized with bitsets.
    
    # Use bitset representation for fast neighbor queries
    # adj_bits[v] is a bitmask of neighbors of v
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # For C5 detection: a C5 exists if there are vertices a,b,c,d,e with
    # a~b, b~c, c~d, d~e, e~a
    # Equivalently: there exist a,c such that they have a common path of length 4
    # but are not adjacent (otherwise it's a triangle with a chord... no)
    # Actually simpler: for each pair (a,c) with a!=c, check if there exist b,d,e
    # with a~b, b~c, c~d, d~e, e~a and b!=d, b!=e, d!=e, a!=b, a!=c, a!=d, a!=e, etc.
    # This is still complex. Let me just enumerate 5-cycles directly.
    # For each ordered 5-tuple (a,b,c,d,e) with all distinct, check the 5 edges.
    # But we can reduce: fix a, then b in N(a), c in N(b)\{a}, d in N(c)\{a,b}, e in N(d)\{a,b,c}
    # then check e~a and e not in {a,b,c,d}.
    
    for a in range(n):
        na = [x for x in range(n) if adj[a][x]]
        for b in na:
            nb = [x for x in range(n) if adj[b][x] and x != a]
            for c in nb:
                nc = [x for x in range(n) if adj[c][x] and x != a and x != b]
                for d in nc:
                    nd = [x for x in range(n) if adj[d][x] and x != a and x != b and x != c]
                    for e in nd:
                        if adj[e][a] and e != d:
                            return True
    return False

def _has_c5_fast(adj_bits, n):
    """Faster C5 check using bit operations."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_bits[a] & (1 << b)):
                continue
            # a~b, need c~b, d~c, e~d, e~a
            common_ab = adj_bits[a] & adj_bits[b]
            # c must be neighbor of b, not a
            for c in range(n):
                if c == a or c == b:
                    continue
                if not (adj_bits[b] & (1 << c)):
                    continue
                # b~c, need d~c, d!=a,b
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    if not (adj_bits[c] & (1 << d)):
                        continue
                    # c~d, need e~d, e~a, e!=a,b,c,d
                    # e must be in N(d) & N(a) \ {a,b,c,d}
                    candidates = adj_bits[d] & adj_bits[a]
                    # Remove a, b, c, d
                    candidates &= ~(1 << a)
                    candidates &= ~(1 << b)
                    candidates &= ~(1 << c)
                    candidates &= ~(1 << d)
                    if candidates:
                        return True
    return False

def _alpha_le_8(adj, n):
    """Check if independence number is <= 8 using greedy + exact for small graphs."""
    # Greedy upper bound: color the complement graph, number of colors = alpha
    # For n <= 35, we can use a branch-and-bound or just check all 9-subsets
    # C(35,9) = 70607460 - too many
    # Use greedy coloring on complement as upper bound
    # Complement adjacency
    comp_adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp_adj[i][j] = 1
                comp_adj[j][i] = 1
    
    # Greedy coloring of complement (which gives upper bound on alpha)
    # Actually, let's use a proper branch and bound for alpha
    # For n<=35, a good B&B should work
    
    # Use the fact that if we can find an independent set of size 9, alpha >= 9
    # Check all 9-subsets? C(35,9) is too large.
    # Use greedy: repeatedly pick a vertex, remove it and its neighbors from complement
    # This gives a lower bound on alpha (a specific independent set)
    # For upper bound, use coloring.
    
    # Let me use a simpler approach: try to find an independent set of size 9
    # via backtracking with pruning
    
    def find_ind_set_9(vertices, size_needed):
        """Try to find independent set of size size_needed from vertices."""
        if size_needed == 0:
            return True
        if len(vertices) < size_needed:
            return False
        # Pruning: if even taking all remaining can't fill size_needed
        # Greedy: pick vertex with fewest connections in induced subgraph
        # Sort by degree in complement (i.e., number of non-neighbors in original)
        # A vertex with few non-neighbors is "hard" to add to independent set
        # Actually, for independent set in G, we want vertices with no edges between them
        # In complement, we want a clique of size 9
        
        # Convert to: find clique of size size_needed in complement induced by vertices
        v_list = list(vertices)
        # Sort by degree in complement (higher degree first for clique finding)
        v_list.sort(key=lambda v: sum(comp_adj[v][u] for u in v_list if u != v), reverse=True)
        
        if not v_list:
            return False
        
        # Pick first vertex, branch: include or exclude
        pivot = v_list[0]
        
        # Include pivot: need clique of size size_needed-1 from neighbors of pivot in complement
        neighbors_in_comp = [u for u in v_list[1:] if comp_adj[pivot][u]]
        if len(neighbors_in_comp) >= size_needed - 1:
            if find_ind_set_9(neighbors_in_comp, size_needed - 1):
                return True
        
        # Exclude pivot
        if find_ind_set_9(v_list[1:], size_needed):
            return True
        
        return False
    
    return not find_ind_set_9(list(range(n)), 9)

def _build_adj(edges, n):
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with 8 copies of K4 (n=32, known valid)
    base_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append([base + i, base + j])
    
    best_n = 32
    best_edges = base_edges[:]
    
    # Try to extend to n=33, 34, 35 by adding vertices with appropriate edges
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 30:
            break
        
        # Add new vertex(es) and try to connect them
        # Strategy: add vertex, connect to some existing vertices, check constraints
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            
            random.seed(attempt * 1000 + target_n)
            
            # Start from base 32-vertex graph
            n = target_n
            edges = base_edges[:]
            adj = _build_adj(edges, n)
            
            # For new vertices (32, 33, 34), decide connections
            new_verts = list(range(32, n))
            
            for nv in new_verts:
                # Connect to a random subset of existing vertices
                # Must avoid creating C5
                # Start with no edges, add one by one
                candidates = list(range(32))
                random.shuffle(candidates)
                
                # Try adding edges from nv to candidates
                for u in candidates:
                    if time.time() > deadline - 30:
                        break
                    # Check if adding edge (nv, u) creates a C5
                    adj[nv][u] = 1
                    adj[u][nv] = 1
                    
                    # Quick C5 check
                    adj_bits = [0] * n
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                adj_bits[i] |= (1 << j)
                                adj_bits[j] |= (1 << i)
                    
                    if _has_c5_fast(adj_bits, n):
                        adj[nv][u] = 0
                        adj[u][nv] = 0
                        continue
                    else:
                        edges.append([nv, u])
            
            # Now check independence number
            if time.time() > deadline - 20:
                break
            
            if _alpha_le_8(adj, n):
                # Valid!
                best_n = n
                best_edges = edges[:]
                break
            else:
                # Try different random connections
                continue
    
    if time.time() > deadline - 5:
        # Return best found
        pass
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
