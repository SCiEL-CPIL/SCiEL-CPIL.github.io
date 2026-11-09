"""Q3-free graph on 16 vertices - maximize edges via greedy randomized search."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 110.0

def _has_q3_with_edge(adj, u, v):
    """Check if adding edge (u,v) creates a Q3 containing this edge.
    
    In Q3, u and v are in different bipartition classes.
    u has 2 other neighbors in the Q3 (in v's class), v has 2 other 
    neighbors in the Q3 (in u's class).
    
    The Q3 structure: if we label vertices as binary triples, u and v
    differ in exactly 1 bit. The other 6 vertices are determined.
    
    Actually, the cleanest check: enumerate all ways to complete the cube.
    In Q3, given an edge (u,v), the cube is uniquely determined up to 
    the choice of which "direction" the edge goes. 
    
    Let's think of Q3 as vertices {0,1}^3. Edge (u,v) means they differ
    in one coordinate, say coordinate k. Then the cube contains:
    - u, v (differ in coord k)
    - u with coord k flipped = v, u with other coords flipped...
    
    More concretely: if u and v differ in coordinate k, then the 8 cube
    vertices are all combinations of:
    - coord k: 0 or 1 (giving u or v's value)
    - other 2 coords: each 0 or 1
    
    So the 8 vertices are: for each of the 2 other coordinates, take
    the value from u or flip it, combined with u's k-value or v's k-value.
    
    But we don't know the labeling. Instead: the 6 other vertices must be
    such that:
    - 2 of them are "neighbors" of u in the cube (differ from u in 1 coord)
    - 2 of them are "neighbors" of v in the cube
    - 2 are "opposite" (differ from both u and v in 2 coords)
    
    This is complex. Let me just do a direct check with pruning.
    
    Given edge (u,v), a Q3 containing it requires 6 more vertices.
    In the Q3, u has degree 3, so 2 other neighbors besides v.
    v has degree 3, so 2 other neighbors besides u.
    
    Let N_u = neighbors of u in current graph (excluding v)
    Let N_v = neighbors of v in current graph (excluding u)
    
    In the Q3, the 2 other neighbors of u must be in N_u, and the 2 other
    neighbors of v must be in N_v.
    
    If |N_u| < 2 or |N_v| < 2, no Q3 can be formed.
    
    Otherwise, for each pair (a,b) from N_u and (c,d) from N_v, check if
    {u,v,a,b,c,d} plus 2 more vertices forms a Q3. Wait, that's 6 vertices,
    we need 8 total.
    
    Actually: the 8 vertices are {u, v} + 2 from N_u + 2 from N_v + 2 more.
    The 2 more must be adjacent to specific vertices.
    
    Let me just enumerate all 6-subsets of the remaining 14 vertices and
    check if the 8-vertex induced subgraph is Q3. With the degree-3 check
    as a fast filter, this should be fast enough.
    """
    adj_np = np.array(adj)
    
    # Fast filter: u needs at least 3 total neighbors, v needs at least 3
    if adj_np[u].sum() < 3 or adj_np[v].sum() < 3:
        return False
    
    remaining = [x for x in range(N) if x != u and x != v]
    
    for comb in combinations(remaining, 6):
        verts = (u, v) + comb
        # Build 8x8 sub-adjacency
        idx = {vert: i for i, vert in enumerate(verts)}
        sub = np.zeros((8, 8), dtype=bool)
        for i, vi in enumerate(verts):
            for j, vj in enumerate(verts):
                if i != j:
                    sub[i, j] = adj_np[vi, vj]
        # Check 3-regular
        degs = sub.sum(axis=1)
        if degs.min() < 3 or degs.max() > 3:
            continue
        # Check bipartite with 4+4
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        bipartite = True
        while queue:
            node = queue.pop(0)
            for nb in range(8):
                if sub[node, nb]:
                    if color[nb] == -1:
                        color[nb] = 1 - color[node]
                        queue.append(nb)
                    elif color[nb] == color[node]:
                        bipartite = False
                        break
            if not bipartite:
                break
        if not bipartite:
            continue
        part0 = sum(1 for c in color if c == 0)
        if part0 == 4:
            return True
    return False


def construct():
    import numpy as np
    from itertools import combinations
    
    start_time = time.time()
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        random.shuffle(all_edges)
        
        adj = [[0] * N for _ in range(N)]
        edge_set = set()
        edge_list = []
        
        for u, v in all_edges:
            if time.time() - start_time > DEADLINE - 3:
                break
            
            adj[u][v] = 1
            adj[v][u] = 1
            
            if _has_q3_with_edge(adj, u, v):
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                edge_set.add((u, v))
                edge_list.append([u, v])
        
        # Local improvement
        improved = True
        while improved and time.time() - start_time < DEADLINE - 3:
            improved = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - 3:
                    break
                for j in range(i + 1, N):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    if (i, j) not in edge_set:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        if not _has_q3_with_edge(adj, i, j):
                            edge_set.add((i, j))
                            edge_list.append([i, j])
                            improved = True
                        else:
                            adj[i][j] = 0
                            adj[j][i] = 0
        
        if len(edge_list) > best_count:
            best_count = len(edge_list)
            best_edges = edge_list[:]
        
        if time.time() - start_time > DEADLINE - 15:
            break
    
    return best_edges
