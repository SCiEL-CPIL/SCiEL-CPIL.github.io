import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    def numpy_greedy_build(seed_rng):
        """Build a K2,3-free graph using vectorized NumPy common-neighbor checks."""
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            neighbors_u = np.where(adj[u] == 1)[0]
            neighbors_v = np.where(adj[v] == 1)[0]
            
            if len(neighbors_u) > 0:
                if np.max(common[neighbors_u, v]) > 1:
                    continue
            
            if len(neighbors_v) > 0:
                if np.max(common[u, neighbors_v]) > 1:
                    continue
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            for w in neighbors_u:
                if w != v:
                    common[w, v] += 1
                    common[v, w] += 1
            
            for w in neighbors_v:
                if w != u:
                    common[u, w] += 1
                    common[w, u] += 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def can_add_edge(adj, common, u, v):
        """Check if edge (u,v) can be added without violating K2,3-free constraint."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
        if len(neighbors_u) > 0:
            if np.max(common[neighbors_u, v]) > 1:
                return False
        if len(neighbors_v) > 0:
            if np.max(common[u, neighbors_v]) > 1:
                return False
        return True

    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
        adj[u, v] = 1
        adj[v, u] = 1
        
        for w in neighbors_u:
            if w != v:
                common[w, v] += 1
                common[v, w] += 1
        
        for w in neighbors_v:
            if w != u:
                common[u, w] += 1
                common[w, u] += 1

    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
        adj[u, v] = 0
        adj[v, u] = 0
        
        for w in neighbors_u:
            if w != v:
                common[w, v] -= 1
                common[v, w] -= 1
        for w in neighbors_v:
            if w != u:
                common[u, w] -= 1
                common[w, u] -= 1

    def numpy_local_search_with_swaps(edges, max_iters=300):
        """Local search with edge additions, removals, and 1-for-2 swaps."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15.0:
            improved = False
            iters += 1
            
            # Phase 1: Try adding any valid non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                if can_add_edge(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_edge_count += 1
                    improved = True
                    break
            
            # Phase 2: Try 1-for-2 swaps (remove one edge, add two)
            if not improved and current_edge_count > 15:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                for rem_u, rem_v in edge_list[:50]:
                    # Save state
                    adj_save = adj.copy()
                    common_save = common.copy()
                    count_save = current_edge_count
                    
                    # Remove edge (rem_u, rem_v)
                    remove_edge(adj, common, rem_u, rem_v)
                    
                    # Try to add two edges from the freed capacity
                    added = 0
                    for u, v in non_edges:
                        if adj[u, v]:
                            continue
                        if u == rem_u and v == rem_v:
                            continue
                        
                        if can_add_edge(adj, common, u, v):
                            add_edge(adj, common, u, v)
                            added += 1
                            if added == 2:
                                break
                    
                    # Accept if we added 2 edges (net +1) or 1 edge (net 0, but might help future)
                    if added >= 1:
                        current_edge_count = count_save - 1 + added
                        improved = True
                        break
                    else:
                        # Restore
                        adj[:] = adj_save
                        common[:] = common_save
                        current_edge_count = count_save
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    def simulated_annealing(edges, max_iters=500):
        """Simulated annealing with 1-for-3 swap moves for escaping local optima."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        best_count = current_edge_count
        best_adj = adj.copy()
        best_common = common.copy()
        
        temperature = 0.6
        cooling_rate = 0.994
        
        for iters in range(max_iters):
            if time.time() > deadline - 20.0:
                break
            
            # Try different move types
            move_type = rng.choice(['add', 'swap_1_2', 'swap_1_3'])
            
            if move_type == 'add':
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                if not non_edges:
                    continue
                u, v = rng.choice(non_edges)
                
                if can_add_edge(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_edge_count += 1
                    if current_edge_count > best_count:
                        best_count = current_edge_count
                        best_adj = adj.copy()
                        best_common = common.copy()
                else:
                    # Reject invalid add
                    pass
                    
            elif move_type == 'swap_1_2':
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if not edge_list:
                    continue
                rem_u, rem_v = rng.choice(edge_list)
                
                adj_save = adj.copy()
                common_save = common.copy()
                count_save = current_edge_count
                
                remove_edge(adj, common, rem_u, rem_v)
                
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v] and not (u == rem_u and v == rem_v)]
                if not non_edges:
                    adj[:] = adj_save
                    common[:] = common_save
                    current_edge_count = count_save
                    continue
                
                added = 0
                for _ in range(2):
                    if not non_edges:
                        break
                    u, v = rng.choice(non_edges)
                    if can_add_edge(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        added += 1
                        non_edges = [(a, b) for a, b in non_edges if not (a == u and b == v)]
                
                if added >= 1:
                    current_edge_count = count_save - 1 + added
                    if current_edge_count > best_count:
                        best_count = current_edge_count
                        best_adj = adj.copy()
                        best_common = common.copy()
                else:
                    adj[:] = adj_save
                    common[:] = common_save
                    current_edge_count = count_save
                    
            elif move_type == 'swap_1_3':
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if not edge_list:
                    continue
                rem_u, rem_v = rng.choice(edge_list)
                
                adj_save = adj.copy()
                common_save = common.copy()
                count_save = current_edge_count
                
                remove_edge(adj, common, rem_u, rem_v)
                
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v] and not (u == rem_u and v == rem_v)]
                if not non_edges:
                    adj[:] = adj_save
                    common[:] = common_save
                    current_edge_count = count_save
                    continue
                
                added = 0
                for _ in range(3):
                    if not non_edges:
                        break
                    u, v = rng.choice(non_edges)
                    if can_add_edge(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        added += 1
                        non_edges = [(a, b) for a, b in non_edges if not (a == u and b == v)]
                
                if added >= 2:  # Net gain of at least 1
                    current_edge_count = count_save - 1 + added
                    if current_edge_count > best_count:
                        best_count = current_edge_count
                        best_adj = adj.copy()
                        best_common = common.copy()
                else:
                    # Accept with probability based on temperature (Metropolis criterion)
                    delta = (count_save - 1 + added) - count_save  # = added - 1
                    if delta < 0:
                        acceptance_prob = np.exp(delta / temperature)
                        if rng.random() < acceptance_prob:
                            current_edge_count = count_save - 1 + added
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                            current_edge_count = count_save
                    else:
                        current_edge_count = count_save - 1 + added
                        if current_edge_count > best_count:
                            best_count = current_edge_count
                            best_adj = adj.copy()
                            best_common = common.copy()
            
            temperature *= cooling_rate
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u, v]:
                    result.append([u, v])
        return result

    def final_polish(edges):
        """High-precision polish: try all possible single-edge additions and 2-edge swaps."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        continue
                    
                    if can_add_edge(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        current_edge_count += 1
                        improved = True
                        break
                if improved:
                    break
            
            # Try all 1-for-2 swaps
            if not improved:
                for rem_u in range(n):
                    for rem_v in range(rem_u + 1, n):
                        if not adj[rem_u, rem_v]:
                            continue
                        
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        remove_edge(adj, common, rem_u, rem_v)
                        
                        added = 0
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    continue
                                if u == rem_u and v == rem_v:
                                    continue
                                
                                if can_add_edge(adj, common, u, v):
                                    add_edge(adj, common, u, v)
                                    added += 1
                                    if added == 2:
                                        break
                            if added == 2:
                                break
                        
                        if added == 2:
                            current_edge_count += 1
                            improved = True
                            break
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                    if improved:
                        break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    # Run multiple greedy builds with different seeds
    for trial in range(25):
        if time.time() > deadline - 30.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = numpy_greedy_build(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            # Local search with swaps
            improved = numpy_local_search_with_swaps(edges, max_iters=200)
            improved = verify(improved)
            
            # Simulated annealing phase
            if improved is not None:
                annealed = simulated_annealing(improved, max_iters=500)
                annealed = verify(annealed)
                
                # Final polish
                if annealed is not None:
                    polished = final_polish(annealed)
                    polished = verify(polished)
                    if polished is not None and len(polished) > best_e:
                        best_e = len(polished)
                        best = polished
                elif improved is not None and len(improved) > best_e:
                    best_e = len(improved)
                    best = improved
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
