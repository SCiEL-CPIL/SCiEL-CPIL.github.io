"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds
    
    n = P['n']
    
    # Build initial adjacency from path
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True
        
    # Precompute path distances
    def bfs_all_pairs(adj):
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            dist[src][src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[src][v] == -1:
                        dist[src][v] = dist[src][u] + 1
                        queue.append(v)
        return dist

    path_dist = bfs_all_pairs(base_adj)
    
    # Get all potential edges (non-existing)
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not base_adj[u][v]:
                all_edges.append((u, v))
                
    # Helper: Check triangle-free
    def is_triangle_free(adj, u, v):
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    # Helper: Compute diameter
    def get_diameter(adj):
        dist = bfs_all_pairs(adj)
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    # Helper: Check if a set of edges is valid
    def is_valid(added_edges):
        # Check duplicates and existing
        edge_set = set()
        for u, v in added_edges:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if base_adj[a][b]:
                return False
                
        # Check triangle-free
        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            if not is_triangle_free(adj, u, v):
                return False
            adj[u][v] = True
            adj[v][u] = True
            
        # Check diameter
        if get_diameter(adj) > 4:
            return False
            
        return True

    # 1. Greedy Initialization
    def greedy_init():
        adj = [row[:] for row in base_adj]
        selected = []
        # Pairs with dist > 4
        uncovered = set()
        for i in range(n):
            for j in range(i+1, n):
                if path_dist[i][j] > 4:
                    uncovered.add((i, j))
                    
        while uncovered:
            best_edge = None
            best_count = -1
            for u, v in all_edges:
                if adj[u][v]:
                    continue
                if not is_triangle_free(adj, u, v):
                    continue
                
                # Estimate coverage
                count = 0
                for (i, j) in uncovered:
                    d1 = path_dist[i][u] + 1 + path_dist[v][j]
                    d2 = path_dist[i][v] + 1 + path_dist[u][j]
                    if min(d1, d2) <= 4:
                        count += 1
                        
                if count > best_count:
                    best_count = count
                    best_edge = (u, v)
                    
            if best_edge is None or best_count == 0:
                break
                
            u, v = best_edge
            adj[u][v] = True
            adj[v][u] = True
            selected.append([u, v])
            
            # Update uncovered (approximate)
            new_uncovered = set()
            for (i, j) in uncovered:
                d1 = path_dist[i][u] + 1 + path_dist[v][j]
                d2 = path_dist[i][v] + 1 + path_dist[u][j]
                if min(d1, d2) > 4:
                    new_uncovered.add((i, j))
            uncovered = new_uncovered
            
        return selected

    best_solution = greedy_init()
    if not is_valid(best_solution):
        # Fallback if greedy fails validity (shouldn't happen if logic is correct, but safe)
        best_solution = []
        
    best_len = len(best_solution)
    
    # 2. Local Search (Hill Climbing with Restarts)
    # We try to remove edges. If stuck, we try swaps.
    
    def local_search(initial_edges, time_limit):
        current_edges = [e[:] for e in initial_edges]
        current_len = len(current_edges)
        
        # Sort edges for consistent removal attempts?
        # Just iterate.
        
        while time.time() < time_limit:
            improved = False
            
            # Try removing edges
            for i in range(len(current_edges)):
                test_edges = current_edges[:i] + current_edges[i+1:]
                if is_valid(test_edges):
                    current_edges = test_edges
                    current_len -= 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping one edge with another
            # Only if we think it might help (e.g., if we are stuck)
            # This is expensive, so limit attempts or do it less frequently
            
            # Random restarts if stuck?
            # For now, just try a few random swaps
            for _ in range(10):
                if not current_edges:
                    break
                idx1 = random.randint(0, len(current_edges) - 1)
                e1 = current_edges[idx1]
                
                # Pick a random candidate to swap in
                # Must be triangle-free with current_edges minus e1
                adj_test = [row[:] for row in base_adj]
                for u, v in current_edges:
                    if u != e1[0] and v != e1[1] and (u, v) != (e1[1], e1[0]):
                        # Simplified: just add all others
                        pass
                
                # Rebuild adj without e1
                adj_test = [row[:] for row in base_adj]
                for u, v in current_edges:
                    if (u, v) != (e1[0], e1[1]) and (u, v) != (e1[1], e1[0]):
                        adj_test[u][v] = True
                        adj_test[v][u] = True
                        
                valid_swaps = []
                for u, v in all_edges:
                    if adj_test[u][v]:
                        continue
                    if not is_triangle_free(adj_test, u, v):
                        continue
                    valid_swaps.append((u, v))
                    
                if valid_swaps:
                    e2 = random.choice(valid_swaps)
                    new_edges = current_edges[:idx1] + current_edges[idx1+1:] + [[e2[0], e2[1]]]
                    if is_valid(new_edges):
                        # Check if it allows removal later?
                        # For now, accept if same length (neutral move) to escape local min
                        # But we want to minimize length. So only accept if we can remove later.
                        # This is a 2-opt like move.
                        # Let's just accept if it's valid, and hope removal works next iter.
                        current_edges = new_edges
                        improved = True
                        break
                        
            if not improved:
                # Stuck. Try random restart with perturbation?
                # Or just break and let outer loop restart?
                # Since we are in a single LS call, let's just break.
                break
                
        return current_edges

    # Run Local Search with multiple restarts
    num_restarts = 50
    restart_time = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Perturb best solution
        if best_solution:
            # Remove 1-2 random edges and add 1-2 random valid edges
            perturbed = [e[:] for e in best_solution]
            if len(perturbed) > 2:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
                
            # Add random valid edge
            adj_test = [row[:] for row in base_adj]
            for u, v in perturbed:
                if is_triangle_free(adj_test, u, v):
                    adj_test[u][v] = True
                    adj_test[v][u] = True
                    
            candidates = []
            for u, v in all_edges:
                if not adj_test[u][v] and is_triangle_free(adj_test, u, v):
                    candidates.append((u, v))
                    
            if candidates:
                u, v = random.choice(candidates)
                perturbed.append([u, v])
        else:
            perturbed = greedy_init()
            
        # Run LS
        ls_sol = local_search(perturbed, time.time() + restart_time)
        
        if is_valid(ls_sol) and len(ls_sol) < best_len:
            best_solution = ls_sol
            best_len = len(ls_sol)
            
    # Final check and return
    if not is_valid(best_solution):
        # Last resort: return empty if invalid? No, must be valid.
        # Try greedy again?
        best_solution = greedy_init()
        
    # Ensure types are correct
    result = []
    for u, v in best_solution:
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
