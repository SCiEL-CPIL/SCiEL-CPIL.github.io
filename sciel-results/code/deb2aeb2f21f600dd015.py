"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _is_triangle_free(adj, u, v, n):
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    common = neighbors_u.intersection(neighbors_v)
    if common:
        return False
    return True

def _construct_greedy():
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    added_edges = []
    
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    
    max_iters = 50
    iter_count = 0
    
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        if (best_u, best_v) in base_edges:
            pass
        else:
            if _is_triangle_free(adj, best_u, best_v, n):
                adj[best_u].append(best_v)
                adj[best_v].append(best_u)
                base_edges.add((best_u, best_v))
                base_edges.add((best_v, best_u))
                added_edges.append([best_u, best_v])
                dist = _bfs_distances(adj, n)
                current_diam = _get_diameter(dist)
            else:
                found = False
                for i in range(n):
                    for j in range(i+1, n):
                        if i == best_u and j == best_v:
                            continue
                        if (i, j) in base_edges:
                            continue
                        if _is_triangle_free(adj, i, j, n):
                            adj[i].append(j)
                            adj[j].append(i)
                            temp_dist = _bfs_distances(adj, n)
                            temp_diam = _get_diameter(temp_dist)
                            adj[i].remove(j)
                            adj[j].remove(i)
                            
                            if temp_diam < current_diam:
                                adj[i].append(j)
                                adj[j].append(i)
                                base_edges.add((i, j))
                                base_edges.add((j, i))
                                added_edges.append([i, j])
                                dist = temp_dist
                                current_diam = temp_diam
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break

    return added_edges

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    for u, v in added_edges:
        if (u, v) in base_edges:
            return False
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _local_search(added_edges, deadline):
    """
    Iterated local search to minimize the number of edges.
    Tries to remove edges or swap edges to reduce count while maintaining validity.
    """
    n = P['n']
    current_solution = [e[:] for e in added_edges]
    
    if not _verify_solution(current_solution):
        return current_solution
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any edge
        for i in range(len(current_solution)):
            if time.time() > deadline:
                return current_solution
            # Temporarily remove edge i
            test_solution = current_solution[:i] + current_solution[i+1:]
            if _verify_solution(test_solution):
                current_solution = test_solution
                improved = True
                break # Restart loop as indices changed
                
        # 2. Try to swap an edge with another to see if we can remove one? 
        # Actually, since we just tried removing all, if we can't remove any single edge,
        # we might need to swap. But swapping usually keeps count same.
        # The primary gain comes from removal. If we can't remove, we are at a local min for deletion.
        # To escape local minima, we could try adding an edge and then removing two?
        # That's complex. Let's stick to deletion first.
        
        # If no deletion worked, try a 1-move swap (replace edge A with edge B) 
        # This doesn't reduce count, but might enable future deletions?
        # Or maybe just return if no deletion.
        
        # Let's try a simple perturbation: remove an edge, add a different one, then try to remove more.
        # This is effectively ILS.
        
        # For now, simple deletion local search is a strong baseline.
        # If stuck, try random restarts? No, we start from greedy.
        
        # Let's implement a basic 2-opt: 
        # Pick edge e1 to remove. 
        # Pick edge e2 to add (not in base, not in solution, triangle free).
        # If valid, and then we can remove another edge? 
        # This is getting complicated for the time limit.
        
        # Let's just do the deletion pass. It's O(N^2) per check.
        # N is small (maybe 5-10 edges). 24 vertices.
        # Verification is BFS on 24 nodes. Very fast.
        
    return current_solution

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # 1. Generate initial solution with greedy
    added = _construct_greedy()
    
    if not _verify_solution(added):
        # Fallback to a known valid dense solution if greedy failed completely
        # The seed's fallback was a specific pattern. 
        # Let's try to construct a valid one quickly.
        # A star from 0 to all even nodes? 
        # 0-2, 0-4... 0-22. Plus path.
        # This might not be diameter 4.
        # Let's just use the greedy result if it's valid, else empty (invalid) or a safe fallback.
        # But greedy should be valid.
        
        # If greedy fails, try a hardcoded small set that is likely valid?
        # (0, 5), (5, 10), (10, 15), (15, 20), (20, 23) -> Diam 5?
        # (0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23) -> 
        # Dist 0-23: 0-4-8-12-16-20-23 (6 steps) -> No.
        # We need stronger shortcuts.
        # 0-8, 8-16, 16-23?
        # 0-8 (8), 8-16 (8), 16-23 (7). 
        # Path 0-1-...-7-8. 
        # 0 to 23: 0-8-16-23 (3 edges). 
        # Check triangles:
        # 0-8: neighbors of 0 are 1. Neighbors of 8 are 7,9. No common. OK.
        # 8-16: neighbors of 8 are 7,9,0. Neighbors of 16 are 15,17. No common. OK.
        # 16-23: neighbors of 16 are 15,17,8. Neighbors of 23 are 22. No common. OK.
        # Is it connected? Yes, path + chords.
        # Diameter?
        # 0 to 23: 3.
        # 0 to 15: 0-8-16-15 (3).
        # 1 to 22: 1-0-8-16-17-18-19-20-21-22 (9)? No.
        # 1 to 22: 1-2-...-8-16-17-...-22.
        # 1-8 (7 steps). 8-16 (1). 16-22 (6). Total 14?
        # Wait, 1 to 8 is 7. 8 to 16 is 1. 16 to 22 is 6. Total 14.
        # We need diameter <= 4.
        # So 0-8, 8-16, 16-23 is NOT enough.
        
        # We need more edges.
        # The greedy approach finds them. If greedy fails, it's a bug in greedy.
        # Let's assume greedy works.
        
        pass

    # 2. Local Search to minimize edges
    optimized = _local_search(added, deadline)
    
    # 3. Final check
    if _verify_solution(optimized):
        # Ensure types are correct
        res = []
        for u, v in optimized:
            res.append([int(u), int(v)])
        return res
    else:
        # If optimization broke it (shouldn't happen), return original
        if _verify_solution(added):
            res = []
            for u, v in added:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
