import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_triangle_free(adj):
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                if u < v:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def is_3_colorable(adj):
        degrees = [bin(m).count('1') for m in adj]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            neighbors = adj[v]
            while neighbors:
                u_bit = neighbors & -neighbors
                u = u_bit.bit_length() - 1
                neighbors ^= u_bit
                if color[u] != -1:
                    forbidden |= (1 << color[u])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2
    
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        if adj[u] & adj[v]:
            return False
        return True
    
    def greedy_add_edges(adj, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if not can_add_edge(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not is_3_colorable(adj):
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return adj
    
    def local_search(adj, deadline):
        best_adj = adj[:]
        best_count = edge_count(adj)
        current_count = best_count
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Try adding edges
            added = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if not can_add_edge(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not is_3_colorable(adj):
                    current_count += 1
                    added = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if added:
                continue
            
            # Try edge swaps: remove one, add another
            current_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
            random.shuffle(current_edges)
            
            for ru, rv in current_edges:
                if time.time() > deadline:
                    break
                
                # Remove edge
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                count_after_removal = current_count - 1
                
                # Try to add a different edge
                found_better = False
                for u, v in all_edges:
                    if time.time() > deadline:
                        break
                    if (u == ru and v == rv) or (u == rv and v == ru):
                        continue
                    if not can_add_edge(adj, u, v):
                        continue
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if not is_3_colorable(adj):
                        new_count = count_after_removal + 1
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = adj[:]
                            current_count = new_count
                            found_better = True
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                if found_better:
                    break
                
                # If no improvement, restore the removed edge
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                current_count = count_after_removal + 1
        
        return best_adj, best_count
    
    def get_clebsch():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def get_grotzsch():
        adj = [0] * n
        for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for v in range(5, 10):
            adj[10] |= (1 << v)
            adj[v] |= (1 << 10)
        return adj
    
    def random_triangle_free():
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    random.seed(42)
    
    best_adj = None
    best_count = 0
    
    # Try multiple initializations
    initializations = [
        get_clebsch(),
        get_grotzsch(),
    ]
    
    # Add several random triangle-free graphs
    for i in range(5):
        initializations.append(random_triangle_free())
    
    time_per_init = (deadline - start_time - 5.0) / len(initializations)
    
    for init_adj in initializations:
        if time.time() > deadline - 3.0:
            break
        
        # Verify initial graph is valid
        if not is_triangle_free(init_adj):
            continue
        if is_3_colorable(init_adj):
            continue
        
        # Greedy edge addition
        init_adj = greedy_add_edges(init_adj, deadline - 2.0)
        
        # Local search
        local_best, local_count = local_search(init_adj, deadline - 1.0)
        
        if local_count > best_count:
            best_count = local_count
            best_adj = local_best
    
    # Final fallback if no valid solution found
    if best_adj is None:
        best_adj = get_clebsch()
        best_count = edge_count(best_adj)
    
    # Convert to edge list
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
    
    return final_edges
