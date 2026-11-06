import random
import time

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Build adjacency as list of sets for fast operations
    def count_c4_from_adj(adj_sets):
        """Count C4s given adjacency as list of sets."""
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = adj_sets[i] & adj_sets[j]
                c = len(common)
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        return c4_count
    
    def build_initial_graph(seed):
        """Build a random graph with exactly target_edges edges."""
        random.seed(seed)
        edge_set = set()
        all_edges_shuffled = all_edges[:]
        random.shuffle(all_edges_shuffled)
        for e in all_edges_shuffled:
            edge_set.add(e)
            if len(edge_set) == target_edges:
                break
        return edge_set
    
    def edge_set_to_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Simulated annealing
    best_edges_set = None
    best_c4 = float('inf')
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random graph
        current_edges = build_initial_graph(restart * 7 + 13)
        current_adj = edge_set_to_adj(current_edges)
        current_c4 = count_c4_from_adj(current_adj)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges_set = set(current_edges)
        
        # Simulated annealing
        T = 100.0
        cooling = 0.995
        min_T = 0.1
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
            
            # Pick a random edge to remove and a random non-edge to add
            edge_list = list(current_edges)
            if not edge_list:
                break
            remove_u, remove_v = random.choice(edge_list)
            
            # Find a random non-edge
            for _ in range(50):
                add_u = random.randint(0, n - 1)
                add_v = random.randint(0, n - 1)
                if add_u == add_v:
                    continue
                if add_u > add_v:
                    add_u, add_v = add_v, add_u
                if (add_u, add_v) not in current_edges:
                    break
            else:
                continue
            
            # Compute delta C4
            # Remove edge (remove_u, remove_v)
            # The C4s affected are those containing this edge
            # A C4 containing edge (u,v) has the form u-a-v-b-u where a,b are common neighbors of u,v... 
            # Actually, it's easier to just compute the change in C4 count.
            
            # Let's compute delta by checking which C4s are created/destroyed
            # This is expensive. Instead, let's use a different move: swap two edges.
            
            # Alternative: just do a full recomputation every few steps, and use a heuristic for delta
            # For now, let's try a simpler approach: random restarts with greedy improvement
            
            # Break out of SA and do greedy local search instead
            break
        
        # Greedy local search: try to swap edges to reduce C4s
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edge_list = list(current_edges)
            random.shuffle(edge_list)
            
            for remove_u, remove_v in edge_list:
                if time.time() > deadline:
                    break
                
                # Try adding each non-edge and see if C4 decreases
                # This is O(n^2) per edge, too slow. Let's be smarter.
                
                # Compute the number of C4s that use edge (remove_u, remove_v)
                # A C4 uses edge (u,v) if there exist a,b such that u-a-v-b-u is a cycle
                # i.e., a is neighbor of u, b is neighbor of v, a is neighbor of b, and a != v, b != u
                # The number of such C4s is the number of pairs (a,b) where:
                # a in adj[u] \ {v}, b in adj[v] \ {u}, a in adj[b]
                
                common_neighbors_uv = current_adj[remove_u] & current_adj[remove_v]
                # C4s through edge (u,v): for each pair of common neighbors a,b of u and v,
                # we have C4 u-a-v-b-u. But we need a-b to be an edge.
                # Number of C4s containing edge (u,v) = number of edges in the subgraph induced by common_neighbors_uv
                
                c4s_with_edge = 0
                cn_list = list(common_neighbors_uv)
                for idx, a in enumerate(cn_list):
                    for b in cn_list[idx+1:]:
                        if b in current_adj[a]:
                            c4s_with_edge += 1
                
                # If we remove this edge, we lose c4s_with_edge C4s
                # Now find the best edge to add that minimizes new C4s
                best_add = None
                best_new_c4 = float('inf')
                
                # Try a sample of candidate edges
                candidates_sample = []
                for _ in range(100):
                    add_u = random.randint(0, n - 1)
                    add_v = random.randint(0, n - 1)
                    if add_u == add_v:
                        continue
                    if add_u > add_v:
                        add_u, add_v = add_v, add_u
                    if (add_u, add_v) not in current_edges:
                        candidates_sample.append((add_u, add_v))
                
                for add_u, add_v in candidates_sample:
                    # New C4s created by adding (add_u, add_v)
                    # = number of edges in common neighbors of add_u and add_v
                    common_new = current_adj[add_u] & current_adj[add_v]
                    new_c4s = 0
                    cn_new_list = list(common_new)
                    for idx, a in enumerate(cn_new_list):
                        for b in cn_new_list[idx+1:]:
                            if b in current_adj[a]:
                                new_c4s += 1
                    
                    delta = -c4s_with_edge + new_c4s
                    if delta < best_new_c4:
                        best_new_c4 = delta
                        best_add = (add_u, add_v)
                
                if best_add is not None and best_new_c4 < 0:
                    # Apply the swap
                    current_edges.discard((remove_u, remove_v))
                    current_adj[remove_u].discard(remove_v)
                    current_adj[remove_v].discard(remove_u)
                    
                    add_u, add_v = best_add
                    current_edges.add((add_u, add_v))
                    current_adj[add_u].add(add_v)
                    current_adj[add_v].add(add_u)
                    
                    current_c4 += best_new_c4
                    improved = True
                    
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges_set = set(current_edges)
                    
                    break  # Restart the outer loop after a change
        
        # After local search, check if we improved
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges_set = set(current_edges)
    
    # Convert to list of lists
    if best_edges_set is None:
        best_edges = all_edges[:target_edges]
    else:
        best_edges = [[u, v] for u, v in sorted(best_edges_set)]
    
    return best_edges
