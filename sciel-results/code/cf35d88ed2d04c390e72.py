import time
import random
import numpy as np

DEADLINE = 145  # Safety margin from 150s limit

def count_c4_fast(adj):
    """
    Fast C4 count using vectorized numpy operations.
    adj: 31x31 numpy int array
    Returns number of 4-cycles.
    """
    # adj^2[u][v] = number of common neighbors of u and v
    # For non-adjacent pairs, C(common, 2) gives pairs of length-2 paths.
    # Each C4 is counted twice (once for each pair of opposite vertices).
    adj_sq = adj @ adj
    # Mask: 1 where adj is 0 (non-edge) and u != v
    # We need upper triangle to avoid double counting in the sum, 
    # but since we divide by 2 at the end, we can sum over all u<v where adj[u][v]==0.
    
    # Create mask for non-edges (excluding diagonal)
    # adj is symmetric. 
    # We want sum_{u<v, adj[u][v]==0} C(adj_sq[u][v], 2)
    
    # Use upper triangle indices
    n = adj.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    
    # Get common neighbors for these pairs
    commons = adj_sq[iu, ju]
    
    # Only consider non-edges
    mask = adj[iu, ju] == 0
    commons = commons[mask]
    
    # C(k, 2) = k*(k-1)//2
    # Only k>=2 contributes
    contribs = commons * (commons - 1) // 2
    total = np.sum(contribs)
    
    return total // 2

def construct_algebraic():
    """
    Construct a graph based on the affine plane AG(2,5) or similar structure.
    n = 31 = 5^2 + 5 + 1. This is the number of points in PG(2,5) or related to polarity graphs.
    
    A known construction for C4-free or low-C4 graphs in this range involves
    using the incidence structure of a projective plane or a polarity graph.
    
    Let's try a specific construction:
    Vertices 0..24 correspond to points in GF(5)xGF(5).
    Vertices 25..30 are "infinity" or special vertices.
    
    Actually, a simpler approach for 31 vertices:
    Use a regular graph of degree 5 or 6.
    91 edges, 31 vertices. Avg degree 5.87.
    
    Let's try a "blow-up" of a small graph or a known extremal graph.
    The ex(31, C4) = 90. We need 91.
    
    One strong candidate structure is the polarity graph of the projective plane PG(2, q) for q=5.
    The polarity graph of PG(2, q) has q^2+q+1 vertices and q(q+1)/2 * (q^2+q+1) / 2? No.
    The number of edges in the polarity graph of PG(2,q) is q(q^2+q+1)/2.
    For q=5: 5*(25+5+1)/2 = 5*31/2 = 77.5. Not an integer?
    Wait, the polarity graph of PG(2,q) is not regular.
    
    Let's look at the "Brown graph" or similar.
    
    Alternative: Start with a random 6-regular graph (if possible) or mix of 5 and 6.
    31 vertices.
    If all degree 6: 31*6/2 = 93 edges.
    We need 91 edges. So remove 2 edges from a 6-regular graph.
    Or start with 5-regular (77.5 edges - impossible).
    
    Let's construct a 6-regular graph on 31 vertices?
    Sum of degrees = 186. 186/31 = 6.
    A 6-regular graph on 31 vertices exists.
    Number of edges = 93.
    We need 91 edges. So we take a 6-regular graph and remove 2 edges.
    Removing edges reduces C4 count?
    Removing an edge (u,v) destroys all C4s containing (u,v).
    This is a good starting point because regular graphs often minimize C4s for a given edge count (or close to it).
    
    How to generate a 6-regular graph on 31 vertices?
    We can use a circulant graph or a random regular graph.
    Let's try a circulant graph C(31, {1, 2, 3, 4, 5, 6}).
    Vertices 0..30. Edge (i, j) if (i-j) mod 31 is in {1, 2, 3, 4, 5, 6, 25, 26, 27, 28, 29, 30}.
    This is 6-regular.
    Edges: 31 * 6 / 2 = 93.
    
    Let's check C4 count of this circulant graph.
    Then remove 2 edges to get to 91.
    Which edges to remove? The ones that are in the most C4s.
    """
    n = 31
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Generate 6-regular circulant graph
    S = [1, 2, 3, 4, 5, 6]
    edges = []
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    # Currently 93 edges. Need to remove 2.
    # Calculate C4 count for each edge to find which ones to remove.
    # Removing edge (u,v) removes C4s where (u,v) is a side.
    # Number of C4s containing edge (u,v) = number of common neighbors of u and v that form a path u-w-x-v?
    # Actually, a C4 containing edge (u,v) looks like u-v-w-x-u.
    # This requires w to be a neighbor of v (w!=u) and x to be a neighbor of u (x!=v) and w-x edge.
    # Alternatively, count C4s in the graph, then for each edge, count how many C4s it belongs to.
    
    # Let's just calculate the C4 count of the whole graph first.
    c4_count = count_c4_fast(adj)
    
    # To minimize C4s after removing 2 edges, we should remove edges that are in the most C4s.
    # Let's estimate the "C4 load" of each edge.
    # For edge (u,v), the number of C4s containing it is the number of paths of length 3 from u to v that do not use (u,v).
    # Path u-a-b-v.
    # a is neighbor of u (a!=v), b is neighbor of v (b!=u), and (a,b) is an edge.
    
    edge_load = {}
    edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]==1]
    
    for u, v in edge_list:
        count = 0
        # Neighbors of u
        neigh_u = np.where(adj[u] == 1)[0]
        # Neighbors of v
        neigh_v = np.where(adj[v] == 1)[0]
        
        for a in neigh_u:
            if a == v: continue
            for b in neigh_v:
                if b == u: continue
                if adj[a, b] == 1:
                    # Path u-a-b-v exists.
                    # Does this form a C4?
                    # Vertices u, a, b, v must be distinct.
                    # u != v (given)
                    # a != v (checked)
                    # b != u (checked)
                    # a != b? If a==b, then u-a-v is a triangle, not a C4 side.
                    # Also need a != u and b != v (true by definition of neighbor, unless self-loop which we don't have).
                    if a != b:
                        count += 1
        edge_load[(u, v)] = count
    
    # Sort edges by load
    sorted_edges = sorted(edge_load.items(), key=lambda x: x[1], reverse=True)
    
    # Remove the top 2 edges
    to_remove = [sorted_edges[0][0], sorted_edges[1][0]]
    
    for u, v in to_remove:
        adj[u, v] = 0
        adj[v, u] = 0
        edges = [e for e in edges if not (e[0]==u and e[1]==v) and not (e[0]==v and e[1]==u)]
        
    return edges, adj

def local_search(edges, adj, time_limit):
    """
    Fast local search using incremental C4 updates.
    """
    n = 31
    m = len(edges)
    edge_set = set((u, v) if u < v else (v, u) for u, v in edges)
    
    # Precompute common neighbors for all pairs?
    # adj is 31x31.
    # common[u][v] = adj[u] & adj[v]
    
    def get_common(u, v):
        return np.bitwise_and(adj[u], adj[v])
    
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    deadline = time.time() + time_limit
    rng = random.Random(42)
    
    # List of edges for fast random selection
    edge_list = list(edge_set)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edge_list:
            break
        u1, v1 = rng.choice(edge_list)
        
        # Pick a random non-edge to add
        # To speed this up, we can precompute non-edges or just sample
        # 31*30/2 = 465 total pairs. 91 edges. 374 non-edges.
        # Sampling is fast enough.
        while True:
            u2 = rng.randint(0, n-1)
            v2 = rng.randint(0, n-1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if (u2, v2) not in edge_set:
                break
        
        # Perform swap
        # Remove (u1, v1)
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        edge_set.discard((u1, v1))
        edge_list.remove((u1, v1))
        
        # Add (u2, v2)
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        edge_set.add((u2, v2))
        edge_list.append((u2, v2))
        
        # Calculate delta in C4
        # This is tricky to do incrementally in O(1) or O(n) without complex bookkeeping.
        # Given n=31, O(n^2) per step is 31^2 = 961 ops.
        # count_c4_fast is vectorized and likely takes ~1ms?
        # 150s / 1ms = 150,000 swaps. That's a lot.
        # Let's try it.
        
        new_c4 = count_c4_fast(adj)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            # Undo
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            edge_set.add((u1, v1))
            edge_list.append((u1, v1))
            
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            edge_set.discard((u2, v2))
            edge_list.remove((u2, v2))
            
    # Convert best_adj back to edge list
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                best_edges.append([i, j])
                
    return best_edges, best_c4

def main():
    start_time = time.time()
    
    # Phase 1: Algebraic Construction
    edges, adj = construct_algebraic()
    
    # Verify edge count
    if len(edges) != 91:
        # Fallback to random if construction failed
        n = 31
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng = random.Random(1)
        edges = rng.sample(all_pairs, 91)
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
    c4_init = count_c4_fast(adj)
    
    # Phase 2: Local Search
    remaining_time = DEADLINE - (time.time() - start_time) - 2
    if remaining_time > 1:
        edges, c4_final = local_search(edges, adj, remaining_time)
    else:
        c4_final = c4_init
        
    return edges

def construct():
    return main()

if __name__ == "__main__":
    result = construct()
    assert len(result) == 91
    n = 31
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in result:
        adj[u, v] = 1
        adj[v, u] = 1
    c4 = count_c4_fast(adj)
    print(f"C4 count: {c4}")
