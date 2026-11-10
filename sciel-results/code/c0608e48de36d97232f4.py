import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    q = 5
    # Build PG(2, q) points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [0] * n  # bitmask adjacency
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_4cycles_bm(adj_list):
        """
        Count 4-cycles correctly.
        For each unordered pair {i,j}, let k = |N(i) ∩ N(j)|.
        Each 4-cycle has exactly 2 pairs of opposite vertices.
        Total 4-cycles = (1/2) * sum over all pairs {i,j} of C(k, 2)
        """
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = adj_list[i] & adj_list[j]
                k = popcount(common)
                if k >= 2:
                    count += k*(k-1)//2
        return count // 2
    
    # Convert to sets for the search
    adj_sets = [set() for _ in range(n)]
    for e in edges:
        u, v = e
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    base_4cycles = count_4cycles_bm(adj)
    
    # Find the best non-edge to add (minimizes new 4-cycles)
    # Using the efficient cost formula: cost = C(common_neighbors, 2) + edge_walks
    best_edge = None
    best_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if j in adj_sets[i]:
                continue
            
            # Cost of adding edge (i,j):
            # 1. New 4-cycles where i,j become opposite: C(|N(i)∩N(j)|, 2)
            common = adj_sets[i] & adj_sets[j]
            k = len(common)
            cost_opposite = k*(k-1)//2
            
            # 2. New 4-cycles where i-j is an edge: count of paths of length 2 from i to j
            #    through a vertex w such that w~i and w~j... wait, that's the same as common neighbors.
            #    Actually, a 4-cycle containing edge (i,j) looks like i-j-a-b-i where a~j, b~i, a~b.
            #    So we need: for each a in N(j)\{i}, for each b in N(i)\{j}, if a~b and a!=b.
            #    Since i,j are not adjacent, a!=i and b!=j are automatic for a in N(j)\{i} and b in N(i)\{j}.
            #    But we also need a!=b.
            cost_edge = 0
            # For efficiency, iterate over the smaller neighborhood
            if len(adj_sets[i]) < len(adj_sets[j]):
                for b in adj_sets[i]:
                    if b == j:
                        continue
                    for a in adj_sets[j]:
                        if a == i or a == b:
                            continue
                        if a in adj_sets[b]:
                            cost_edge += 1
            else:
                for a in adj_sets[j]:
                    if a == i:
                        continue
                    for b in adj_sets[i]:
                        if b == j or b == a:
                            continue
                        if b in adj_sets[a]:
                            cost_edge += 1
            
            total_new = cost_opposite + cost_edge
            
            if total_new < best_cost:
                best_cost = total_new
                best_edge = [i, j]
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    if best_edge is not None:
        u, v = best_edge
        edges.append(best_edge)
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Now do local search: try edge swaps to reduce 4-cycles
    current_count = count_4cycles_bm(adj)
    
    # Build list of edges for easier manipulation
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute non-edges
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i,j))
    
    def compute_swap_cost(e1, e2, adj_local, adj_sets_local):
        """
        Compute the change in 4-cycles when removing e1=(u1,v1) and adding e2=(u2,v2).
        Returns the net change (negative = improvement).
        """
        u1, v1 = e1
        u2, v2 = e2
        
        # We need to compute C4 count before and after, but that's expensive.
        # Instead, let's use the incremental approach.
        # This is complex, so for now we'll just do the full count for accuracy.
        # But that's too slow for many swaps. Let's use a smarter approach.
        
        # Actually, let's just temporarily apply and count.
        # Remove e1
        adj_local[u1] &= ~(1 << v1)
        adj_local[v1] &= ~(1 << u1)
        adj_sets_local[u1].discard(v1)
        adj_sets_local[v1].discard(u1)
        
        # Add e2
        adj_local[u2] |= (1 << v2)
        adj_local[v2] |= (1 << u2)
        adj_sets_local[u2].add(v2)
        adj_sets_local[v2].add(u2)
        
        new_count = count_4cycles_bm(adj_local)
        
        # Revert
        adj_local[u1] |= (1 << v1)
        adj_local[v1] |= (1 << u1)
        adj_sets_local[u1].add(v1)
        adj_sets_local[v1].add(u1)
        
        adj_local[u2] &= ~(1 << v2)
        adj_local[v2] &= ~(1 << u2)
        adj_sets_local[u2].discard(v2)
        adj_sets_local[v2].discard(u2)
        
        return new_count - current_count
    
    # Local search with simulated annealing
    improved = True
    max_iterations = 100
    iteration = 0
    random.seed(42)
    
    # Temperature schedule for simulated annealing
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.95
    
    current_T = T_start
    best_count = current_count
    best_edge_set = set(edge_set)
    
    while iteration < max_iterations:
        iteration += 1
        current_T *= cooling_rate
        
        if time.time() > deadline - 30:
            break
        
        # Generate a random swap
        edge_list = list(edge_set)
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        if not non_edges or not edge_list:
            break
        
        e1 = random.choice(edge_list)
        e2 = random.choice(non_edges)
        
        # Apply swap
        u1, v1 = e1
        u2, v2 = e2
        
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        adj_sets[u1].discard(v1)
        adj_sets[v1].discard(u1)
        
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        adj_sets[u2].add(v2)
        adj_sets[v2].add(u2)
        
        edge_set.remove(e1)
        edge_set.add(e2)
        
        new_count = count_4cycles_bm(adj)
        delta = new_count - current_count
        
        # Accept or reject based on simulated annealing
        if delta < 0 or random.random() < np.exp(-delta / current_T):
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_edge_set = set(edge_set)
        else:
            # Revert
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            adj_sets[u1].add(v1)
            adj_sets[v1].add(u1)
            
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            adj_sets[u2].discard(v2)
            adj_sets[v2].discard(u2)
            
            edge_set.add(e1)
            edge_set.remove(e2)
        
        # Check if we should continue
        if time.time() > deadline - 30:
            break
    
    # Use the best solution found
    if best_count < current_count:
        # Reconstruct from best_edge_set
        adj = [0] * n
        adj_sets = [set() for _ in range(n)]
        for u, v in best_edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    # Final validation
    final_edges = []
    seen = set()
    for u, v in best_edge_set:
        key = (min(u, v), max(u, v))
        if key not in seen:
            final_edges.append([u, v])
            seen.add(key)
    
    # Ensure exactly 91 distinct edges
    if len(final_edges) > 91:
        final_edges = final_edges[:91]
    elif len(final_edges) < 91:
        # Add more edges with minimal 4-cycle increase
        adj_tmp = [0] * n
        for e in final_edges:
            u, v = e
            adj_tmp[u] |= (1 << v)
            adj_tmp[v] |= (1 << u)
        
        candidates = []
        for i, j in all_pairs:
            if (i,j) not in seen:
                candidates.append((i,j))
        
        # Sort by cost
        for i, j in candidates:
            if len(final_edges) >= 91:
                break
            adj_tmp[i] |= (1 << j)
            adj_tmp[j] |= (1 << i)
            new_count = count_4cycles_bm(adj_tmp)
            adj_tmp[i] &= ~(1 << j)
            adj_tmp[j] &= ~(1 << i)
            
            final_edges.append([i, j])
            seen.add((i,j))
            adj_tmp[i] |= (1 << j)
            adj_tmp[j] |= (1 << i)
        
        # If still not enough, just add arbitrary
        for i in range(n):
            if len(final_edges) >= 91:
                break
            for j in range(i+1, n):
                if len(final_edges) >= 91:
                    break
                if (i,j) not in seen:
                    final_edges.append([i,j])
                    seen.add((i,j))
    
    return final_edges[:91]
