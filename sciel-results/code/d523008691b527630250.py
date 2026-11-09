"""K3,3-free dense graph on 20 vertices via greedy with random restarts."""
import time
import random

def construct():
    """
    Construct a K3,3-free simple graph on vertices 0..19 maximizing edges.
    
    Strategy: Greedy edge addition with random restarts.
    For each pair of vertices, check if adding the edge would create a K3,3.
    An edge (u,v) creates a K3,3 iff there exist three vertices a,b,c such that
    {a,b,c} are all common neighbors of some pair {x,y} where {x,y} is disjoint
    from {u,v} and the edge connects u to v completing the bipartition.
    
    More precisely: adding edge (u,v) creates a K3,3 iff there exists a set of
    3 vertices S and 3 vertices T such that:
    - u in S, v in T (or u in T, v in S)
    - Every vertex in S is adjacent to every vertex in T (after adding u-v)
    
    Equivalent check: adding (u,v) is forbidden iff there exist vertices a,b,c
    (all distinct from u,v) such that:
    - a,b,c are all neighbors of u AND all neighbors of v (common neighbors)
      -> then {u,a,b,c} x {v, ...} no wait.
    
    Let me reconsider. A K3,3 has two parts of size 3. If we add edge (u,v),
    then u and v must be in different parts. So we need:
    - 3 vertices in part A containing u
    - 3 vertices in part B containing v
    - All cross-edges present
    
    So: there exist a,b (distinct from u,v) and c,d (distinct from u,v, and c,d != a,b)
    such that:
    - a,b are neighbors of u, and c,d are neighbors of v
    - a,b are neighbors of c,d (i.e., a-c, a-d, b-c, b-d all exist)
    - c,d are neighbors of u (u-c, u-d exist)
    - a,b are neighbors of v (v-a, v-b exist)
    
    This is expensive to check naively. Better approach:
    
    For adding edge (u,v):
    Let CN = common neighbors of u and v (excluding each other).
    If |CN| >= 3, then pick any 3 from CN: {a,b,c}. Then we need a 4th vertex... 
    wait no. If u and v are in different parts of K3,3, and a,b,c are common neighbors,
    then {u,a,b} x {v,c,...} no that doesn't work either.
    
    Let me think more carefully. K3,3-free means no induced or subgraph K3,3.
    The graph is K3,3-free iff for every pair of disjoint 3-sets {a,b,c} and {d,e,f},
    not all 9 cross-edges are present.
    
    When adding edge (u,v), the only new cross-edges that could complete a K3,3
    are those involving u-v. So we need to check: is there a K3,3 where u and v
    are in different parts AND the edge u-v is one of the cross edges?
    
    If u is in part A = {u, x, y} and v is in part B = {v, z, w}, then we need:
    - x,y in N(u) ∩ N(v) (common neighbors of u and v)
    - z,w in N(u) ∩ N(v) (common neighbors of u and v)  
    Wait, that's not right either.
    
    Part A = {u, x, y}, Part B = {v, z, w}.
    Cross edges needed: u-v, u-z, u-w, x-v, x-z, x-w, y-v, y-z, y-w.
    
    So x and y must be adjacent to u and v (in N(u) ∩ N(v)).
    z and w must be adjacent to u and v (in N(u) ∩ N(v)).
    Additionally, x-z, x-w, y-z, y-w must all be present.
    
    So: Let CN = N(u) ∩ N(v). We need to find 4 distinct vertices x,y,z,w in CN
    such that x-z, x-w, y-z, y-w are all edges. Equivalently, in the subgraph induced
    by CN, we need a K2,2 (a 4-cycle) where {x,y} and {z,w} form the two parts.
    
    Actually simpler: we need x,y,z,w in CN with all four edges xz, xw, yz, yw present.
    This means {x,y} and {z,w} form a K2,2 in G[CN].
    
    So the check is: does G[CN] contain a K2,2? I.e., are there 4 vertices in CN
    forming a 4-cycle with the right bipartition?
    
    Even simpler: pick any two vertices x,y in CN. If they share at least 2 common
    neighbors in CN (other than themselves), then we have a K2,2.
    
    Let me implement this efficiently with bitsets.
    """
    n = 20
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def check_edge_valid(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # Find common neighbors of u and v
        cn_mask = adj[u] & adj[v]
        cn_mask &= ~(1 << u)
        cn_mask &= ~(1 << v)
        
        # In the subgraph induced by CN, check if there's a K2,2
        # A K2,2 exists iff there are x,y in CN with |N(x) ∩ N(y) ∩ CN| >= 2
        cn_list = []
        m = cn_mask
        while m:
            lowbit = m & (-m)
            i = lowbit.bit_length() - 1
            cn_list.append(i)
            m ^= lowbit
        
        # For each pair in cn_list, check common neighbors within cn_list
        L = len(cn_list)
        for i in range(L):
            for j in range(i + 1, L):
                common = adj[cn_list[i]] & adj[cn_list[j]] & cn_mask
                # Count bits
                if common.bit_count() >= 2:
                    return False
        return True
    
    best_edges = None
    best_count = 0
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random order of edges
        edges_order = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges_order)
        
        # Adjacency as list of bitmasks
        adj = [0] * n
        edge_set = set()
        
        for u, v in edges_order:
            if time.time() > deadline:
                break
            if check_edge_valid(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edge_set]
    
    # Try a few more restarts if we have time
    while time.time() < deadline - 5:
        edges_order = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges_order)
        adj = [0] * n
        edge_set = set()
        
        for u, v in edges_order:
            if check_edge_valid(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edge_set]
    
    if best_edges is None:
        # Fallback: K2,18
        best_edges = [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges
