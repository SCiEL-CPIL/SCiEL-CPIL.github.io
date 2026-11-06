import time
import random
from itertools import combinations

def is_2_colorable_fast(edges, n):
    """
    Fast 2-colorability check using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Build vertex-to-edge index
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For each edge, store as a tuple for fast access
    edge_tuples = [tuple(e) for e in edges]
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    # Precompute neighbors for faster propagation? 
    # The current propagation is O(5 * deg(v)) per step.
    
    def propagate(v, c):
        """Assign color c to vertex v with unit propagation. Returns False on contradiction."""
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        # Process all edges containing v
        for e_idx in vertex_edges[v]:
            e = edge_tuples[e_idx]
            count_c = 0
            count_1c = 0
            unassigned = -1
            
            for u in e:
                cu = colors[u]
                if cu == c:
                    count_c += 1
                elif cu == 1 - c:
                    count_1c += 1
                else:
                    unassigned = u
            
            # Contradiction check
            if count_c == 5 or count_1c == 5:
                return False
            
            # Forced assignment
            if count_c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            elif count_1c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, c):
                        return False
        
        return True
    
    def backtrack():
        # Find unassigned vertex with highest degree
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True  # All assigned, no contradictions
        
        # Try color 0
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Best known: 51 edges.
    # We will try to find a hypergraph with <= 51 edges.
    # If we can't find 51, we return the best we found.
    
    best_count = 1000
    best_edges = []
    best_n = 0
    
    # Try different n values
    # Known minimal is 22 vertices, 51 edges.
    # Let's try n from 18 to 25.
    
    for n in range(18, 26):
        if time.time() > deadline - 10.0:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try multiple random initializations
        max_trials = 20 if n == 22 else 10
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            # Start with a random subset of edges
            # To ensure it's non-2-colorable, we might need many edges initially.
            # Let's start with 100 edges.
            current_edges = [list(e) for e in all_edges[:100]]
            
            # Ensure it is non-2-colorable
            # If it is 2-colorable, add more edges until it's not.
            idx = 100
            while is_2_colorable_fast(current_edges, n):
                if idx >= len(all_edges):
                    break
                current_edges.append(list(all_edges[idx]))
                idx += 1
                if time.time() > deadline - 5.0:
                    break
            
            if is_2_colorable_fast(current_edges, n):
                continue # Failed to make it non-2-colorable
                
            # Now we have a non-2-colorable hypergraph.
            # Peel edges to minimize count.
            # We try to remove edges one by one.
            # To be efficient, we shuffle the order of removal.
            remove_order = list(range(len(current_edges)))
            random.shuffle(remove_order)
            
            # We iterate through the removal order.
            # If removing edge i keeps it non-2-colorable, we remove it.
            # We repeat this process until no more edges can be removed.
            
            changed = True
            while changed:
                changed = False
                # Shuffle removal order again for better mixing?
                random.shuffle(remove_order)
                
                for i in remove_order:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # If this edge was already removed, skip
                    # We need to track which edges are present.
                    # Easier: maintain a list of present edges.
                    # Let's use a set of tuples for fast lookup?
                    # No, list of lists is fine if we manage indices carefully.
                    
                    # This simple "try remove i" approach is O(N^2 * CheckTime).
                    # N=100, CheckTime is small.
                    
                    # Try removing current_edges[i]
                    # We need to check if the remaining graph is non-2-colorable.
                    
                    # Optimization: Only try removing if the edge is "critical"?
                    # Hard to know. Just try.
                    
                    # To avoid O(N^2) complexity with high N, we can just iterate through the list.
                    # But we are shuffling.
                    
                    # Let's just iterate through a fixed list of indices present in the current graph.
                    # This is getting complicated with dynamic list modification.
                    
                    # Alternative: 
                    # Keep a list `present`.
                    # For each edge in `present` (shuffled), try to remove it.
                    # If successful, remove it and restart the loop?
                    # Or just continue to next edge.
                    
                    pass # The logic above was messy. Let's rewrite the peeling loop.
            
            # Rewrite Peeling Logic
            present = [list(e) for e in current_edges]
            changed = True
            while changed:
                changed = False
                # Shuffle indices
                indices = list(range(len(present)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Try removing present[idx]
                    # Create a new list without this edge
                    # To be efficient, we can just check the subgraph.
                    
                    # If len(present) is small, this is fast.
                    # If len(present) is large, it's slow.
                    # We only do this if we are close to the deadline.
                    
                    # Let's only peel if len(present) > best_count
                    if len(present) <= best_count:
                        break # No point peeling if we can't beat the best
                    
                    # Check if removing present[idx] leaves it non-2-colorable
                    # We do this by checking the graph with that edge removed.
                    
                    # Efficient check:
                    # We can just call is_2_colorable_fast on the list without that edge.
                    # To avoid list copying, we can pass a list and a "skip" index?
                    # My is_2_colorable_fast doesn't support skip.
                    # List copying is O(N). N is ~50-100. This is fine.
                    
                    test_edges = [e for j, e in enumerate(present) if j != idx]
                    
                    if not is_2_colorable_fast(test_edges, n):
                        # Removal successful
                        present.pop(idx)
                        changed = True
                        # Restart the loop to try removing more from the new set
                        break # Break the for loop, restart while loop
                
            if len(present) < best_count:
                best_count = len(present)
                best_edges = present
                best_n = n
                
            if best_count <= 51:
                return {'n': best_n, 'edges': best_edges}

    return {'n': best_n, 'edges': best_edges}
