import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Try to find a graph for n=46, then 45, then 44
    for n in [46, 45, 44]:
        min_deg_req = n - 39
        
        # Try multiple random restarts
        for attempt in range(20000):
            if time.time() > deadline:
                break
                
            # Initialize graph
            adj = [set() for _ in range(n)]
            
            # Greedy add edges to satisfy min degree
            success = True
            while True:
                # Check if min degree requirement is met
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_req:
                    break
                
                # Find a vertex with minimum degree
                min_vertex = -1
                min_d = n + 1
                # Randomize the order of checking to avoid bias
                indices = list(range(n))
                random.shuffle(indices)
                for i in indices:
                    if len(adj[i]) < min_d:
                        min_d = len(adj[i])
                        min_vertex = i
                
                # Try to add an edge to min_vertex
                found = False
                # Candidates: vertices not adjacent to min_vertex
                # To ensure we don't miss easy connections, we can sort by degree or just shuffle
                candidates = [v for v in range(n) if v != min_vertex and v not in adj[min_vertex]]
                random.shuffle(candidates)
                
                for v in candidates:
                    # Check if adding (min_vertex, v) creates a C4
                    # A C4 is min_vertex - v - w - x - min_vertex
                    # This requires w in N(v) and x in N(min_vertex) and (w,x) in E
                    # Also w != x
                    creates_c4 = False
                    # Iterate over neighbors of min_vertex
                    # If N(min_vertex) is small, this is fast
                    for x in adj[min_vertex]:
                        # Check if any neighbor of x is in N(v)
                        # And that neighbor is not x (which would be a triangle, allowed)
                        # Actually, if w is in N(x) and w is in N(v), then min_vertex-x-w-v-min_vertex is a C4?
                        # Edges: (min_vertex, x), (x, w), (w, v).
                        # Add (min_vertex, v).
                        # Cycle: min_vertex - v - w - x - min_vertex.
                        # Yes.
                        # So check if N(x) intersects N(v) excluding x?
                        # If w = x, then min_vertex-x-v is a path of length 2. Adding min_vertex-v makes a triangle.
                        # Triangles are allowed.
                        # So we only care if there is w in N(x) such that w in N(v) and w != x.
                        for w in adj[x]:
                            if w in adj[v] and w != x:
                                creates_c4 = True
                                break
                        if creates_c4:
                            break
                    
                    if not creates_c4:
                        adj[min_vertex].add(v)
                        adj[v].add(min_vertex)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                # Construct edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append([u, v])
                return {'n': n, 'edges': edge_list}
            
    # Fallback: return the trivial solution if no better found
    return {'n': 39, 'edges': []}
