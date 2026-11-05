import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Helper: Check if adding edge (u, v) creates a triangle
    def can_add_edge(adj, u, v):
        if adj[u][v]:
            return False
        # Check for common neighbor
        for w in range(16):
            if adj[u][w] and adj[v][w]:
                return False
        return True

    # Helper: Add edge to adj
    def add_edge(adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    # Helper: Get list of edges from adj
    def get_edges(adj):
        edges = []
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Helper: Check 3-colorability
    def is_3_colorable(adj):
        colors = [-1] * 16
        
        def backtrack(vertex):
            if vertex == 16:
                return True
            
            # MRV Heuristic: Pick the uncolored vertex with the fewest available colors
            # Simple version: Just iterate 0..15
            for c in range(3):
                # Check if color c is valid
                valid = True
                for u in range(16):
                    if adj[vertex][u] and colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    colors[vertex] = -1
            
            return False
        
        return backtrack(0)

    # Function to construct the base Grötzsch graph (Mycielski of C5) on vertices 0..10
    def get_grotzsch_edges():
        edges = []
        # Cycle 0-1-2-3-4-0
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([2, 3])
        edges.append([3, 4])
        edges.append([4, 0])
        
        # Copies 5-1, 6-2, 7-3, 8-4, 9-0
        edges.append([5, 1])
        edges.append([6, 2])
        edges.append([7, 3])
        edges.append([8, 4])
        edges.append([9, 0])
        
        # Apex 10 connected to 5,6,7,8,9
        edges.append([10, 5])
        edges.append([10, 6])
        edges.append([10, 7])
        edges.append([10, 8])
        edges.append([10, 9])
        
        return edges

    best_count = 0
    best_edges = []
    
    base_edges = get_grotzsch_edges()
    
    # We will try to add vertices 11, 12, 13, 14, 15
    # Strategy: Randomized Greedy with MIS
    # For a new vertex v, we can connect it to any Independent Set in the current graph.
    # To maximize edges, we connect it to a Large Independent Set.
    
    def find_max_independent_set(adj, vertices):
        # vertices: list of vertex indices to consider
        # Returns a list of vertices forming an independent set
        # Greedy heuristic: Sort by degree (descending) or random
        # For better results, we can try multiple random orderings
        # Here we use a simple greedy: pick a random vertex, remove it and its neighbors
        # Repeat until empty.
        
        # To make it more effective, we'll do a few restarts inside this function?
        # No, keep it simple but randomized outside.
        
        # Actually, a better heuristic for "Maximize edges added by v" is to pick the MIS of the induced subgraph.
        # Since the graph is small, we can use a simple backtracking for MIS if needed, 
        # but greedy is fast.
        
        # Let's use a simple greedy:
        # 1. Create a set of available vertices from 'vertices'
        # 2. While available is not empty:
        #    - Pick a random vertex u from available
        #    - Add u to the independent set
        #    - Remove u and all its neighbors (within 'vertices') from available
        
        available = set(vertices)
        independent_set = []
        
        while available:
            # Pick random vertex
            u = random.choice(list(available))
            independent_set.append(u)
            # Remove u and neighbors
            to_remove = {u}
            for v in available:
                if adj[u][v]:
                    to_remove.add(v)
            available -= to_remove
            
        return independent_set

    random.seed(42)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        adj = [[0]*16 for _ in range(16)]
        
        # Add base edges
        for u, v in base_edges:
            add_edge(adj, u, v)
        
        # Vertices to add: 11, 12, 13, 14, 15
        new_vertices = [11, 12, 13, 14, 15]
        random.shuffle(new_vertices)
        
        # Add each new vertex
        for v in new_vertices:
            # Potential neighbors are all existing vertices 0..v-1
            # We can connect v to any Independent Set of the subgraph induced by 0..v-1
            # Note: The subgraph induced by 0..v-1 is triangle-free.
            
            candidates = list(range(v))
            
            # Find a large independent set in the subgraph induced by candidates
            # We want to maximize the number of edges, so we want the largest independent set.
            # Greedy MIS with randomization
            # To get variety, we can run the greedy MIS a few times and pick the largest
            best_mis = []
            for _ in range(10): # 10 random restarts for MIS
                mis = find_max_independent_set(adj, candidates)
                if len(mis) > len(best_mis):
                    best_mis = mis
            
            # Connect v to all vertices in best_mis
            for u in best_mis:
                add_edge(adj, v, u)
        
        # Verify
        if not is_3_colorable(adj):
            edges = get_edges(adj)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                
        # Optional: Try to improve by local search?
        # For now, rely on the randomized construction.

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
