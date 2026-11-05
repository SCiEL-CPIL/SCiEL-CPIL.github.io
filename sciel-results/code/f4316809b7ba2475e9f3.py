"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using efficient local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin for safety

    # We want to maximize n. Best known is 17.
    # We will try to find a valid coloring for n=17, then 16, etc.
    
    # Pre-compute edge mappings for max n=20
    # For a given n, we need to map index -> (u, v)
    
    def get_edge_map(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def solve_for_n(n, time_limit):
        if n <= 1:
            return {'n': n, 'colors': []}
        
        edges = get_edge_map(n)
        num_edges = len(edges)
        
        # Map from edge index to (u, v)
        edge_to_uv = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_uv[idx] = (u, v)
            
        # Map from (u, v) to edge index (for quick lookup if needed, but we iterate indices)
        
        def count_c6s_containing_edge(adj, n, u, v):
            """
            Count the number of simple 6-cycles in the graph defined by adj
            that contain the edge (u, v).
            adj is a list of sets of neighbors.
            """
            # We need paths of length 5 from u to v in G - (u,v)
            # Path: u - a - b - c - d - v
            count = 0
            # Ensure u and v are neighbors in adj (they should be, as we are counting C6s in this color class)
            # If they are not neighbors, count is 0.
            if v not in adj[u]:
                return 0
            
            # Iterate over possible 'a'
            for a in adj[u]:
                if a == v:
                    continue
                # a must be distinct from u, v
                # Iterate over possible 'b'
                for b in adj[a]:
                    if b == u or b == v or b == a:
                        continue
                    # Iterate over possible 'c'
                    for c in adj[b]:
                        if c == u or c == v or c == a or c == b:
                            continue
                        # Iterate over possible 'd'
                        for d in adj[c]:
                            if d == u or d == v or d == a or d == b or d == c:
                                continue
                            # Check if d is connected to v
                            if v in adj[d]:
                                count += 1
            return count

        def total_c6_count(colors, adj):
            """
            Calculate total number of C6s.
            Note: Each C6 is counted 6 times (once for each edge in the cycle) 
            if we sum count_c6s_containing_edge for all edges.
            So total_c6s = sum / 6.
            But for local search, we only care about relative changes.
            We can maintain the sum of counts over all edges.
            """
            total = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                total += count_c6s_containing_edge(adj[c], n, u, v)
            return total // 6

        # Local Search
        best_result = None
        
        while time.time() < time_limit:
            # Random restart
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Build initial adjacency lists
            adj = [set() for _ in range(4)]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c].add((u, v))
                adj[c].add((v, u)) # Wait, adj is list of sets of neighbors.
                # Correction: adj[c] should be a list of sets, where adj[c][i] is set of neighbors of i.
            
            # Re-initialize adj properly
            adj = [set() for _ in range(4)] # This is wrong, need list of lists of sets
            adj = [ [set() for _ in range(n)] for _ in range(4) ]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u].add(v)
                adj[c][v].add(u)

            # Calculate initial total C6 count (sum of counts per edge)
            # We store the sum of counts over all edges to make updates easy.
            # Sum = sum_{e} count_c6s_containing_edge(e)
            # Total C6s = Sum / 6
            current_sum = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                current_sum += count_c6s_containing_edge(adj[c], n, u, v)
            
            # If current_sum == 0, we found a solution
            if current_sum == 0:
                return {'n': n, 'colors': colors}

            # Local Search Loop
            no_improvement = 0
            max_no_improve = 5000 # Limit per restart
            
            while no_improvement < max_no_improve and time.time() < time_limit:
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                old_color = colors[idx]
                
                # Check if this edge is in any C6
                # If count_c6s_containing_edge(adj[old_color], n, u, v) == 0, 
                # changing it might not help reduce the sum immediately, 
                # but it might break a C6 that uses it? No, if it's 0, it's not in a C6.
                # So we should prefer edges that are in a C6.
                
                # Optimization: Only pick edges that are in a C6.
                # To do this efficiently, we could maintain a set of bad edges.
                # But for now, let's just pick randomly and check.
                # If it's not bad, we might waste time.
                # Let's try a few random edges and pick the first one that is "bad".
                
                bad_idx = -1
                for _ in range(10):
                    trial_idx = random.randint(0, num_edges - 1)
                    tu, tv = edge_to_uv[trial_idx]
                    tc = colors[trial_idx]
                    if count_c6s_containing_edge(adj[tc], n, tu, tv) > 0:
                        bad_idx = trial_idx
                        break
                
                if bad_idx == -1:
                    # Couldn't find a bad edge in 10 tries?
                    # If current_sum > 0, there must be one.
                    # Just pick a random one.
                    bad_idx = random.randint(0, num_edges - 1)
                
                idx = bad_idx
                u, v = edge_to_uv[idx]
                old_color = colors[idx]
                
                # Try all other colors
                best_new_color = old_color
                best_new_sum = current_sum
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    # Calculate delta
                    # Remove old_color contribution
                    old_count = count_c6s_containing_edge(adj[old_color], n, u, v)
                    # Add new_color contribution
                    # We need to simulate adding (u,v) to adj[new_color]
                    # But count_c6s_containing_edge expects the edge to be present in adj.
                    # So we temporarily add it, count, then remove.
                    
                    # Temporarily add
                    adj[new_color][u].add(v)
                    adj[new_color][v].add(u)
                    new_count = count_c6s_containing_edge(adj[new_color], n, u, v)
                    # Temporarily remove
                    adj[new_color][u].remove(v)
                    adj[new_color][v].remove(u)
                    
                    delta = new_count - old_count
                    new_sum = current_sum + delta
                    
                    if new_sum < best_new_sum:
                        best_new_sum = new_sum
                        best_new_color = new_color
                
                if best_new_color != old_color:
                    # Apply change
                    colors[idx] = best_new_color
                    # Update adj
                    adj[old_color][u].remove(v)
                    adj[old_color][v].remove(u)
                    adj[best_new_color][u].add(v)
                    adj[best_new_color][v].add(u)
                    
                    # Update current_sum
                    old_count = count_c6s_containing_edge(adj[old_color], n, u, v) # This is now 0? No, we removed it.
                    # Wait, we calculated old_count BEFORE removing.
                    # We need to re-calculate or store it.
                    # Let's re-calculate the sum from scratch if we are worried about bugs.
                    # But incremental is faster.
                    # Let's trust the delta.
                    current_sum = best_new_sum
                    no_improvement = 0
                    
                    if current_sum == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improvement += 1
                    # Simulated Annealing: accept worse moves with some probability?
                    # For now, just continue.
        
        return None

    # Try to find solution for decreasing n
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        res = solve_for_n(n, deadline)
        if res:
            return res
            
    # Fallback to n=9 known solution
    n = 9
    colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
