import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit
    
    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)
    
    # Verify a graph has no K5 and no independent 5-set
    def is_valid(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                return False
        return True

    # Optimized Local Search
    def try_find(n, max_attempts=3, max_steps=50000, seed_base=0):
        if n < 5:
            return n, []
            
        # Precompute 5-sets
        all_5sets = list(itertools.combinations(range(n), 5))
        num_5sets = len(all_5sets)
        
        # Map edge to list of 5-set indices
        edge_to_5sets = {}
        # Map 5-set index to list of edges (tuples)
        five_set_edges = [None] * num_5sets
        
        for idx, combo in enumerate(all_5sets):
            edges_in_set = []
            for i in range(5):
                for j in range(i + 1, 5):
                    e = (combo[i], combo[j])
                    edges_in_set.append(e)
                    if e not in edge_to_5sets:
                        edge_to_5sets[e] = []
                    edge_to_5sets[e].append(idx)
            five_set_edges[idx] = edges_in_set
            
        all_edges = list(edge_to_5sets.keys())
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 10:
                break
                
            random.seed(seed_base + attempt)
            
            # Initialize with random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Initialize edge_count for 5-sets
            # edge_count[idx] = number of edges present in all_5sets[idx]
            edge_count = [0] * num_5sets
            viol_k5 = 0
            viol_i5 = 0
            
            for idx, combo in enumerate(all_5sets):
                count = 0
                for i in range(5):
                    u = combo[i]
                    for j in range(i + 1, 5):
                        v = combo[j]
                        if adj[u] & (1 << v):
                            count += 1
                edge_count[idx] = count
                if count == 10:
                    viol_k5 += 1
                elif count == 0:
                    viol_i5 += 1
                    
            current_viol = viol_k5 + viol_i5
            best_viol = current_viol
            best_adj = adj[:]
            
            # Local search
            for step in range(max_steps):
                if time.time() - start_time > timeout - 10:
                    break
                    
                if current_viol == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append((i, j))
                    return n, edges
                    
                # Pick random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                e = (i, j)
                affected_indices = edge_to_5sets.get(e, [])
                
                is_present = bool(adj[i] & (1 << j))
                
                # Calculate delta violations
                delta_viol = 0
                # If we add edge (is_present is False):
                #   count increases by 1.
                #   If count was 0 (i5), it becomes 1 (no i5). delta_i5 = -1.
                #   If count was 9 (k4+1), it becomes 10 (k5). delta_k5 = +1.
                # If we remove edge (is_present is True):
                #   count decreases by 1.
                #   If count was 10 (k5), it becomes 9 (no k5). delta_k5 = -1.
                #   If count was 1 (i4+1), it becomes 0 (i5). delta_i5 = +1.
                
                for idx in affected_indices:
                    c = edge_count[idx]
                    if not is_present:
                        # Adding edge
                        if c == 0:
                            delta_viol -= 1 # i5 removed
                        elif c == 9:
                            delta_viol += 1 # k5 created
                    else:
                        # Removing edge
                        if c == 10:
                            delta_viol -= 1 # k5 removed
                        elif c == 1:
                            delta_viol += 1 # i5 created
                            
                new_viol = current_viol + delta_viol
                
                # Accept if better or with small probability
                if new_viol < current_viol or (new_viol == current_viol and random.random() < 0.5) or (new_viol > current_viol and random.random() < 0.1):
                    # Apply flip
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    
                    # Update edge_counts
                    for idx in affected_indices:
                        if is_present:
                            edge_count[idx] -= 1
                        else:
                            edge_count[idx] += 1
                            
                    current_viol = new_viol
                    
                    if current_viol < best_viol:
                        best_viol = current_viol
                        best_adj = adj[:]
                        
                else:
                    # Revert (do nothing as we didn't apply)
                    pass
                    
            # End local search loop
            if best_viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
                
        return None, None

    # Try to find 42-vertex graph, then 41, 40, etc.
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() - start_time > timeout - 30:
            break
            
        n_found, edges_found = try_find(target_n, max_attempts=3, max_steps=50000, seed_base=target_n * 100)
        
        if n_found == target_n and edges_found is not None:
            if is_valid(target_n, edges_found):
                return target_n, edges_found
                
    # Fallback: return the Paley 37
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
