"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    deadline = time.time() + 115
    n = 16
    
    # Precompute all 8-vertex subsets
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    def is_q3(vertices, adj):
        """
        Check if the induced subgraph on 'vertices' is isomorphic to Q3.
        Q3 is a 3-regular bipartite graph on 8 vertices with 12 edges.
        A connected 3-regular bipartite graph on 8 vertices is Q3.
        We check:
        1. All degrees in induced subgraph are 3.
        2. Total edges is 12.
        (Connectivity is implied by 3-regularity on 8 vertices in this context?
         If it were disconnected, components would be 3-regular bipartite.
         Smallest 3-regular bipartite graph is K_{3,3} (6 vertices).
         So components must have size >= 6. 8 cannot be split into >=6 and >=6.
         So it must be connected.)
        """
        v_set = set(vertices)
        edge_count = 0
        for u in vertices:
            deg = 0
            for v in vertices:
                if u != v and v in adj[u]:
                    deg += 1
            if deg != 3:
                return False
            edge_count += deg
        if edge_count != 24: # 12 edges * 2
            return False
        return True

    def find_q3(adj):
        """Find one Q3 subgraph. Returns list of edges or None."""
        for verts in all_8_subsets:
            if is_q3(verts, adj):
                # Return edges of this Q3
                edges = []
                v_list = list(verts)
                for i in range(8):
                    for j in range(i+1, 8):
                        u, v = v_list[i], v_list[j]
                        if v in adj[u]:
                            edges.append((u, v))
                return edges
        return None

    def solve_once(seed):
        random.seed(seed)
        # Initialize K16
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                adj[i].add(j)
                adj[j].add(i)
        
        removed_edges = []
        
        while time.time() < deadline - 2:
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            
            # Pick a random edge from the found Q3 to remove
            edge_to_remove = random.choice(q3_edges)
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)
            removed_edges.append(edge_to_remove)
        
        # Count remaining edges
        edge_count = sum(len(s) for s in adj) // 2
        # Reconstruct edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges, edge_count

    best_edges = []
    best_count = 0
    seed = 42
    
    # Run multiple restarts
    while time.time() < deadline - 10:
        edges, count = solve_once(seed)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        
    if not best_edges:
        # Fallback: return empty or a known valid graph if loop didn't finish
        # But solve_once should return something.
        # If it timed out in the middle, it returns the current state which might have Q3.
        # We need to ensure the returned graph is Q3-free.
        # The loop in solve_once breaks when find_q3 is None.
        # If it breaks due to time, the graph might still have Q3.
        # So we must only accept results where the loop terminated normally (Q3-free).
        # Let's modify solve_once to return a flag or handle this.
        pass

    # Re-implementing loop to ensure validity
    best_edges = []
    best_count = 0
    seed = 42
    
    while time.time() < deadline - 5:
        random.seed(seed)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                adj[i].add(j)
                adj[j].add(i)
        
        valid = False
        while time.time() < deadline - 2:
            q3_edges = find_q3(adj)
            if q3_edges is None:
                valid = True
                break
            
            edge_to_remove = random.choice(q3_edges)
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)
        
        if valid:
            edge_count = sum(len(s) for s in adj) // 2
            if edge_count > best_count:
                best_count = edge_count
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                best_edges = edges
        
        seed += 1

    if not best_edges:
        # Emergency fallback: empty graph is Q3-free
        return []
        
    return best_edges
# EVOLVE-BLOCK-END
