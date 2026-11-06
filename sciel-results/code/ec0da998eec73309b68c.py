import time
import random
import math

def count_c6_using_edge(adj, u, v):
    """
    Counts the number of C6 cycles in the graph defined by adj 
    that use the edge (u, v).
    This is equivalent to counting simple paths of length 5 from u to v 
    in the graph with edge (u, v) removed.
    adj is a list of sets, adj[i] contains neighbors of i.
    """
    # We need to find paths u -> x1 -> x2 -> x3 -> x4 -> v
    # such that all vertices are distinct and edges exist.
    # We can do DFS from u, avoiding v until the last step, 
    # and avoiding u and previously visited nodes.
    
    # Optimization: Since we are removing (u,v), we just don't traverse it.
    # But adj already represents the graph state. 
    # If we are calculating the delta for ADDING (u,v) to G_c', 
    # then adj is G_c' WITHOUT (u,v).
    # So we just count paths of length 5 from u to v in adj.
    
    count = 0
    # Depth first search
    # path: list of vertices
    # We need depth 5.
    # u is start. v is target.
    
    # To speed up, we can prune.
    # If we are at depth d, we have used d+1 vertices.
    # We need to reach v at depth 5.
    
    def dfs(current, target, depth, visited):
        nonlocal count
        if depth == 5:
            if current == target:
                count += 1
            return
        
        next_depth = depth + 1
        for neighbor in adj[current]:
            if neighbor == current:
                continue
            if neighbor in visited:
                continue
            # Pruning: if neighbor is target and next_depth < 5, we can't go there yet?
            # Actually, if neighbor is target, we can only accept if next_depth == 5.
            # If next_depth < 5 and neighbor == target, we can skip because we can't form a simple path 
            # that hits target early and then leaves it (simple path).
            # Wait, a simple path cannot visit a vertex twice. So if we hit target early, 
            # we can't continue. So if next_depth < 5 and neighbor == target, skip.
            if neighbor == target and next_depth < 5:
                continue
            
            visited.add(neighbor)
            dfs(neighbor, target, next_depth, visited)
            visited.remove(neighbor)

    visited = {u}
    dfs(u, v, 0, visited)
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    def is_valid_coloring(n, colors):
        # colors is a list of length n*(n-1)//2
        # Reconstruct adjacency lists for each color
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        # Check for C6 in each color
        for c in range(4):
            # Build adjacency list for this color
            adj_list = [set() for _ in range(n)]
            for (u, v) in adj[c]:
                adj_list[u].add(v)
                adj_list[v].add(u)
            
            # Check all edges for C6 involvement? 
            # Or just check if any C6 exists.
            # If any C6 exists, it uses some edge (u,v).
            # So if for any edge (u,v) in color c, count_c6_using_edge(adj_list, u, v) > 0, then invalid.
            # But count_c6_using_edge counts paths of length 5. 
            # A C6 using (u,v) corresponds to a path of length 5.
            # So we just need to check if there is ANY C6.
            # We can iterate over all edges (u,v) in color c and check if count > 0.
            # This might be redundant but safe.
            # Actually, if a C6 exists, it has 6 edges. It will be counted 6 times (once for each edge).
            # So if we find any edge with count > 0, we have a C6.
            
            for (u, v) in adj[c]:
                # We need to pass the adjacency list of the graph WITHOUT (u,v) to count_c6_using_edge?
                # No, count_c6_using_edge(adj, u, v) as defined above assumes adj is the graph 
                # and we are looking for paths in adj.
                # But the path cannot use (u,v) itself.
                # My DFS implementation in count_c6_using_edge does not explicitly forbid using (u,v) 
                # unless we remove it.
                # In the delta calculation, we will pass the graph WITHOUT the edge.
                # For the initial check, we should remove (u,v) from adj_list temporarily?
                # Or modify count_c6_using_edge to take a 'forbidden_edge' parameter.
                
                # Let's modify count_c6_using_edge to be robust.
                # Actually, for the initial check, we can just use the same logic.
                # Let's create a helper that checks if a specific graph has a C6.
                
                pass # We'll implement a direct check function below
                
        return True

    def has_c6_in_graph(adj_list, n):
        # adj_list is list of sets
        # Check if there is any C6
        # Iterate over all edges
        for u in range(n):
            for v in adj_list[u]:
                if u < v: # check each edge once
                    # Check if there is a path of length 5 from u to v in G \ {(u,v)}
                    # We can do this by temporarily removing (u,v) or just ignoring it in DFS
                    # Let's use a DFS that ignores (u,v)
                    
                    # To avoid creating a new adj_list, we can pass a 'skip_edge' to DFS
                    # Or just create a copy? n is small.
                    # Let's create a modified check
                    
                    # Actually, we can just use the count_c6_using_edge logic 
                    # but we need to ensure it doesn't use (u,v).
                    
                    # Let's refactor count_c6_using_edge to take adj and a forbidden edge
                    pass

    # Refactored Count
    def count_c6_using_edge_with_skip(adj_list, u, v, skip_u, skip_v):
        count = 0
        def dfs(current, target, depth, visited):
            nonlocal count
            if depth == 5:
                if current == target:
                    count += 1
                return
            
            next_depth = depth + 1
            for neighbor in adj_list[current]:
                if neighbor == current:
                    continue
                if neighbor in visited:
                    continue
                # Skip the edge (skip_u, skip_v)
                if (current == skip_u and neighbor == skip_v) or (current == skip_v and neighbor == skip_u):
                    continue
                if neighbor == target and next_depth < 5:
                    continue
                
                visited.add(neighbor)
                dfs(neighbor, target, next_depth, visited)
                visited.remove(neighbor)

        visited = {u}
        dfs(u, v, 0, visited)
        return count

    def is_valid_coloring(n, colors):
        adj_sets = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_sets[c].add((i, j))
                idx += 1
        
        for c in range(4):
            adj_list = [set() for _ in range(n)]
            for (u, v) in adj_sets[c]:
                adj_list[u].add(v)
                adj_list[v].add(u)
            
            for (u, v) in adj_sets[c]:
                cnt = count_c6_using_edge_with_skip(adj_list, u, v, u, v)
                if cnt > 0:
                    return False
        return True

    def get_cost(n, colors):
        # Returns total number of C6s
        adj_sets = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_sets[c].add((i, j))
                idx += 1
        
        total_cost = 0
        for c in range(4):
            adj_list = [set() for _ in range(n)]
            for (u, v) in adj_sets[c]:
                adj_list[u].add(v)
                adj_list[v].add(u)
            
            for (u, v) in adj_sets[c]:
                cnt = count_c6_using_edge_with_skip(adj_list, u, v, u, v)
                # Each C6 is counted 6 times (once for each edge)
                total_cost += cnt
        return total_cost // 6

    def find_valid(n, time_limit):
        # Simulated Annealing
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build initial state
        adj_sets = [set() for _ in range(4)]
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_sets[c].add((i, j))
                edge_idx[(i, j)] = idx
                idx += 1
        
        current_cost = get_cost(n, colors)
        if current_cost == 0:
            return colors
        
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        start = time.time()
        
        while time.time() < start + time_limit and T > min_T:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            # Find (u,v) for e_idx
            # We can precompute a list of edges
            # Let's just iterate to find it? No, precompute.
            # This is inefficient. Let's store edges in a list.
            
            # Optimization: Precompute edge list
            break # Placeholder, need to restructure
            
        return None

    # Restructure for speed: Precompute edges
    def find_valid_fast(n, time_limit):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        num_edges = len(edges)
        
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Adjacency sets for each color
        adj_sets = [set() for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj_sets[c].add((u, v))
            
        # Build adj lists for each color
        adj_lists = []
        for c in range(4):
            adj = [set() for _ in range(n)]
            for (u, v) in adj_sets[c]:
                adj[u].add(v)
                adj[v].add(u)
            adj_lists.append(adj)
        
        current_cost = get_cost(n, colors)
        if current_cost == 0:
            return colors
            
        T = 10.0
        cooling = 0.99995
        min_T = 0.01
        start = time.time()
        
        while time.time() < start + time_limit and T > min_T:
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            if random.random() < 0.5:
                continue # Skip if same? No, pick a different color.
            
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Calculate Delta Cost
            # Remove (u,v) from old_c
            # Add (u,v) to new_c
            
            # Count C6s lost in old_c
            # These are C6s in G_old that use (u,v)
            # = count_c6_using_edge_with_skip(adj_lists[old_c], u, v, u, v)
            lost = count_c6_using_edge_with_skip(adj_lists[old_c], u, v, u, v)
            
            # Count C6s gained in new_c
            # These are C6s in G_new U {(u,v)} that use (u,v)
            # = count_c6_using_edge_with_skip(adj_lists[new_c], u, v, u, v)
            # Note: adj_lists[new_c] does NOT contain (u,v) yet.
            gained = count_c6_using_edge_with_skip(adj_lists[new_c], u, v, u, v)
            
            delta = gained - lost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                # Update structures
                adj_sets[old_c].remove((u, v))
                adj_sets[new_c].add((u, v))
                
                adj_lists[old_c][u].remove(v)
                adj_lists[old_c][v].remove(u)
                
                adj_lists[new_c][u].add(v)
                adj_lists[new_c][v].add(u)
                
                colors[idx] = new_c
                current_cost += delta
                
                if current_cost == 0:
                    return colors
            
            T *= cooling
            
        return None

    # Try n=18
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for target_n in [18, 19, 17]:
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        print(f"Trying n={target_n}, time left: {remaining_time}")
        result = find_valid_fast(target_n, remaining_time - 2)
        if result is not None:
            # Verify
            if is_valid_coloring(target_n, result):
                return {'n': target_n, 'colors': result}
            else:
                print(f"n={target_n} result failed verification")
                
    return {'n': best_n, 'colors': best_colors}
