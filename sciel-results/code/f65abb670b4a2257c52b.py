import random
import time

def construct():
    n = 16
    # Set seed for reproducibility
    random.seed(42)
    
    # 1. Initialize with Grötzsch Graph (11 vertices) + 5 isolated vertices
    # This is a known 4-chromatic triangle-free graph.
    adj = [set() for _ in range(n)]
    edges = []
    
    # Grötzsch Graph edges
    # Vertices 0-4: C5
    # Vertices 5-9: Copies
    # Vertex 10: Root
    grotzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 5], [2, 6], [3, 7], [4, 8], # Wait, standard Mycielski mapping:
        # Let's use the standard construction to be safe.
        # C5: 0-1-2-3-4-0
        # Mycielski: 5-9 correspond to 0-4? No, usually 5..9 are copies.
        # Let's just hardcode a known valid Grötzsch graph edge list.
        # Reference: Grötzsch graph edges (11 vertices)
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],
        [0, 10], [1, 10], [2, 10], [3, 10], [4, 10]
    ]
    # Wait, the Mycielski of C5:
    # V1: 0..4 (C5)
    # V2: 5..9 (copies)
    # V3: 10 (root)
    # Edges:
    # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copies connected to original neighbors:
    # 5 connected to N(0) = {1, 4} -> (5,1), (5,4)
    # 6 connected to N(1) = {0, 2} -> (6,0), (6,2)
    # 7 connected to N(2) = {1, 3} -> (7,1), (7,3)
    # 8 connected to N(3) = {2, 4} -> (8,2), (8,4)
    # 9 connected to N(4) = {3, 0} -> (9,3), (9,0)
    # Copies form a C5? No, copies are not connected to each other in Mycielski.
    # Root 10 connected to all copies: (10,5), (10,6), (10,7), (10,8), (10,9)
    
    # Let's re-verify the standard Grötzsch graph edges.
    # It is the Mycielski of C5.
    # C5: 0-1-2-3-4-0
    # Copies 5,6,7,8,9.
    # 5 is copy of 0. Neighbors of 0 are 1,4. So 5-1, 5-4.
    # 6 is copy of 1. Neighbors of 1 are 0,2. So 6-0, 6-2.
    # 7 is copy of 2. Neighbors of 2 are 1,3. So 7-1, 7-3.
    # 8 is copy of 3. Neighbors of 3 are 2,4. So 8-2, 8-4.
    # 9 is copy of 4. Neighbors of 4 are 3,0. So 9-3, 9-0.
    # Root 10 connected to 5,6,7,8,9.
    
    correct_grotzsch = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 4],
        [6, 0], [6, 2],
        [7, 1], [7, 3],
        [8, 2], [8, 4],
        [9, 3], [9, 0],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    for u, v in correct_grotzsch:
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        
    # Vertices 11-15 are isolated.
    
    def is_triangle_free():
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True

    def is_3_colorable():
        # Backtracking 3-coloring
        colors = [-1] * n
        def backtrack(i):
            if i == n:
                return True
            for c in range(3):
                ok = True
                for neighbor in adj[i]:
                    if neighbor < i and colors[neighbor] == c:
                        ok = False
                        break
                if ok:
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
            return False
        return backtrack(0)
        
    # Check initial state
    assert is_triangle_free()
    assert not is_3_colorable()
    
    current_edges = len(edges)
    edge_set = set(frozenset((u, v)) for u, v in edges)
    
    # Simulated Annealing
    start_temp = 1.0
    end_temp = 0.01
    cooling_rate = 0.99
    max_iterations = 20000
    start_time = time.time()
    timeout = 110 # 120s limit, leave buffer
    
    best_edges = edges[:]
    best_edge_set = edge_set.copy()
    best_count = current_edges
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        temp = start_temp * (cooling_rate ** i)
        
        # Choose a move: Add edge or Remove edge
        # Bias towards adding edges since we want to maximize
        if random.random() < 0.8:
            # Try to add a random edge
            # Find all non-edges
            # Optimization: Precompute possible edges?
            # Just pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if frozenset((u, v)) in edge_set:
                continue
            
            # Check triangle constraint
            # Adding (u, v) creates triangle if there is w in N(u) intersect N(v)
            common = adj[u] & adj[v]
            if common:
                continue # Reject, triangle created
            
            # Accept move (Add edge)
            # New graph is supergraph, so if old was not 3-colorable, new is not.
            # So we don't need to check 3-colorability for additions!
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(frozenset((u, v)))
            edges.append([u, v])
            current_edges += 1
            
            if current_edges > best_count:
                best_count = current_edges
                best_edges = edges[:]
                best_edge_set = edge_set.copy()
                
        else:
            # Try to remove a random edge
            if not edges:
                continue
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard(frozenset((u, v)))
            edges.pop(idx)
            current_edges -= 1
            
            # Check if still not 3-colorable
            # If it becomes 3-colorable, we must reject (or accept with penalty, but we want to maximize edges of a valid graph)
            # The task says "construct() -> ... that is not 3-colourable".
            # So the final output must be not 3-colorable.
            # If we remove an edge and it becomes 3-colorable, the state is invalid for the final answer.
            # We can revert immediately or accept and revert later.
            # Better to revert immediately if it becomes 3-colorable, as we lose the "4-chromatic" property.
            # Wait, is it possible to go through a 3-colorable state to reach a better non-3-colorable state?
            # Yes, but it's risky. Simpler: only accept removals if the graph remains non-3-colorable.
            if is_3_colorable():
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add(frozenset((u, v)))
                edges.append([u, v])
                current_edges += 1
            else:
                # Accept removal (valid state)
                # Update best if we somehow improved? No, removal decreases edges.
                # But it might allow future additions that lead to a higher total.
                pass

    return best_edges
