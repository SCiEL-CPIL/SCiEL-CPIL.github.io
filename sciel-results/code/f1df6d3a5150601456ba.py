import time
import random
from itertools import combinations

# Deadline guard: must finish well before 150s wall-clock
_START_TIME = time.time()
DEADLINE = 120.0  # safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Return coloured edges [[u, v, c], ...] of a simple graph on vertices 0..19
    with colours 0..19 such that every 4-cycle is rainbow.
    Maximize the number of edges.
    
    Strategy: Construct a C4-free graph. A graph with no 4-cycles satisfies the 
    rainbow condition vacuously. We use a greedy algorithm to build a maximal 
    C4-free subgraph of K_20.
    """
    n = 20
    # Use a fixed seed for reproducibility, or random if we want to explore
    # Greedy order can affect the final number of edges.
    random.seed(42)
    
    # All possible edges in K_20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Shuffle edges to potentially get a denser C4-free graph
    # Different orderings can lead to different maximal C4-free graphs
    random.shuffle(all_edges)
    
    # Adjacency matrix for fast lookup
    # adj[u][v] = True if edge (u,v) is in the graph
    adj = [[False] * n for _ in range(n)]
    
    # List of edges in the current graph
    graph_edges = []
    
    # To check if adding (u, v) creates a C4:
    # A C4 involving (u, v) would look like u-x-y-v-u.
    # This exists if there is a path of length 2 from u to v in the current graph.
    # i.e., there exists x, y such that (u,x), (x,y), (y,v) are in the graph.
    # Actually, simpler: if there is any common neighbor of u and v? No.
    # If u and v have a common neighbor w, then u-w-v is a path of length 2.
    # Adding (u,v) creates a triangle, not a C4.
    # A C4 is u-x-y-v-u.
    # So we need to check if there is a path u-x-y-v.
    # This is equivalent to checking if the distance between u and v is 2?
    # No, if distance is 2, say u-x-v, adding (u,v) makes a triangle.
    # If distance is 3, say u-x-y-v, adding (u,v) makes a C4.
    # So we need to ensure that there is no path of length 3 between u and v.
    
    # Efficient check:
    # For each neighbor x of u, and each neighbor y of x, if y is a neighbor of v,
    # and y != u and x != v, then we have u-x-y-v.
    # Wait, x is neighbor of u. y is neighbor of x. If y is neighbor of v, we have u-x-y-v.
    # We must ensure the vertices are distinct: u, x, y, v.
    # Since (u,v) is not in the graph yet, u != v.
    # x is neighbor of u, so x != u.
    # y is neighbor of x, so y != x.
    # y is neighbor of v, so y != v.
    # We also need x != v (otherwise u-v-v? No, x=v implies u-v edge exists? No, u-v not in graph. But x is neighbor of u. If x=v, then (u,v) is in graph. Contradiction. So x != v is guaranteed if (u,v) not in graph).
    # We also need y != u. If y=u, then x is neighbor of u and u is neighbor of x (symmetric). And u is neighbor of v. So u-x-u-v? No, simple path.
    # If y=u, then we have u-x-u-v. This is not a simple cycle.
    # So we just need to check if there exists x in N(u) and y in N(x) such that y in N(v) and y != u.
    
    def creates_c4(u, v):
        # Check if adding (u, v) creates a C4
        # N(u) = neighbors of u
        # For each x in N(u):
        #   For each y in N(x):
        #     If y in N(v) and y != u:
        #        Return True
        # Note: We iterate x in N(u).
        # We can optimize by precomputing neighbors lists.
        pass

    # Let's maintain neighbor lists for speed
    neighbors = [[] for _ in range(n)]
    
    def has_path_len_3(u, v):
        # Check if there is a path u-x-y-v
        # x must be neighbor of u
        # y must be neighbor of x
        # y must be neighbor of v
        # y != u
        for x in neighbors[u]:
            for y in neighbors[x]:
                if y == v:
                    continue # y must be distinct from v? No, y is intermediate. v is endpoint.
                # Actually, the path is u-x-y-v.
                # So y must be adjacent to v.
                # Check if y is in neighbors[v]
                # Using a set for neighbors[v] would be faster, but list is small (max 19)
                if y in neighbors[v] and y != u:
                    return True
        return False

    for u, v in all_edges:
        if _time_left() < 10.0:
            break
            
        if not has_path_len_3(u, v):
            # Add edge (u, v)
            adj[u][v] = True
            adj[v][u] = True
            neighbors[u].append(v)
            neighbors[v].append(u)
            graph_edges.append((u, v))
            
    # The resulting graph is C4-free.
    # Color all edges with color 0.
    # This is a valid coloring because there are no 4-cycles to violate the rainbow condition.
    
    # Ensure we return the correct format [[u, v, c], ...]
    result = [[u, v, 0] for u, v in graph_edges]
    
    # Optional: Verify validity for safety (should be fast for ~50 edges)
    # But given the construction logic, it's safe.
    
    return result
