import random
import itertools

def construct():
    n = 20
    num_colors = 20
    best_edges = []
    best_count = 0
    
    # Precompute all 4-cycles? No, check on the fly.
    # Or precompute all potential 4-cycles (sets of 4 vertices)
    # There are C(20,4) = 4845 sets of 4 vertices.
    # For each set, there are 3 ways to form a C4.
    
    # We will use a greedy approach:
    # 1. Generate a random permutation of all possible edges in K_20.
    # 2. For each edge (u,v), try to assign a color.
    # 3. A color c is valid if it doesn't create a bad C4.
    #    A bad C4 is a cycle u-v-w-x-u where edges (v,w), (w,x), (x,u) exist.
    #    The colors of (v,w), (w,x), (x,u) must be distinct from c and each other.
    
    # To make it fast, we maintain the graph as a set of edges (u,v,c).
    # For a new edge (u,v,c), we need to check all paths of length 3 from u to v.
    # Path: u - x - w - v.
    # Edges: (u,x), (x,w), (w,v).
    # Colors: c_u_x, c_x_w, c_w_v.
    # We need c != c_u_x, c != c_x_w, c != c_w_v.
    # And c_u_x, c_x_w, c_w_v must be distinct?
    # The condition is that the 4 edges of the C4 have distinct colors.
    # The other 3 edges might already be in a bad C4? No, we assume the current graph is valid.
    # So the other 3 edges are part of some valid C4s, but their mutual colors are not the issue for the NEW cycle, 
    # UNLESS the new edge closes a cycle where the other 3 edges are not mutually distinct?
    # Wait. The condition is "Every 4-cycle ... must receive four distinct colours".
    # If the current graph is valid, any existing C4 has distinct colors.
    # When we add (u,v,c), we create new C4s that include (u,v).
    # For each such new C4 (u,v,w,x), the other 3 edges (v,w), (w,x), (x,u) are already in the graph.
    # They might not have distinct colors among themselves?
    # If they didn't, the graph would have been invalid already?
    # No, (v,w), (w,x), (x,u) do not form a C4 by themselves. They form a path.
    # They might be part of other C4s.
    # But for the NEW C4, we just need the 4 colors to be distinct.
    # So we need:
    # 1. c != color(v,w)
    # 2. c != color(w,x)
    # 3. c != color(x,u)
    # 4. color(v,w) != color(w,x)
    # 5. color(w,x) != color(x,u)
    # 6. color(v,w) != color(x,u)
    
    # Conditions 4,5,6 depend only on the existing graph.
    # If the existing graph has a path v-w-x-u where the 3 edges have non-distinct colors, 
    # then adding ANY color to (u,v) will result in a C4 with non-distinct colors (since 3 are already bad).
    # So we should only add (u,v) if there exists a color c such that c is distinct from the 3 edge colors, 
    # AND the 3 edge colors are mutually distinct.
    
    # Algorithm:
    # 1. Randomize edge order.
    # 2. For each edge (u,v):
    #    - Find all paths of length 3 from u to v: u-x-w-v.
    #    - For each path, check if colors of (u,x), (x,w), (w,v) are mutually distinct.
    #    - If yes, collect the set of forbidden colors {c_u_x, c_x_w, c_w_v}.
    #    - If no, this path blocks the addition of (u,v) entirely? 
    #      Yes, because if the 3 existing edges have a collision, any 4th color will result in a non-rainbow C4.
    #      Wait, if c_u_x == c_x_w, then the new C4 has at least 2 same colors. Invalid.
    #      So if ANY path of length 3 has non-distinct colors among its 3 edges, we CANNOT add (u,v).
    #    - If all paths have mutually distinct edge colors, then we just need to pick c not in the union of all forbidden colors from all paths.
    #    - If such c exists, add edge (u,v,c).
    
    # Complexity:
    # Number of edges ~ 190.
    # For each edge, number of paths of length 3.
    # In dense graph, many paths.
    # 190 * (number of paths) * (check colors).
    # This is feasible.
    
    def try_construct(seed):
        random.seed(seed)
        edges = {} # (u,v) -> color
        adj = {i: set() for i in range(n)}
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            forbidden = set()
            valid_path_exists = True # If no paths, it's valid (no C4 created)
            
            # Find all x, w such that (u,x), (x,w), (w,v) exist
            # u - x - w - v
            for x in adj[u]:
                for w in adj[x]:
                    if w == u or w == v: continue
                    if v in adj[w]:
                        # Path u-x-w-v found
                        c1 = edges.get((min(u,x), max(u,x)))
                        c2 = edges.get((min(x,w), max(x,w)))
                        c3 = edges.get((min(w,v), max(w,v)))
                        
                        # Check if c1, c2, c3 are distinct
                        if len({c1, c2, c3}) < 3:
                            valid_path_exists = False
                            break
                            
                        forbidden.add(c1)
                        forbidden.add(c2)
                        forbidden.add(c3)
                if not valid_path_exists:
                    break
            if not valid_path_exists:
                continue
                
            # Find a color not in forbidden
            for c in range(num_colors):
                if c not in forbidden:
                    # Add edge
                    key = (min(u,v), max(u,v))
                    edges[key] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    break
        
        return list([[u, v, c] for (u,v), c in edges.items()])

    # Run multiple times
    for seed in range(100):
        res = try_construct(seed)
        if len(res) > best_count:
            best_count = len(res)
            best_edges = res
            
    return best_edges
