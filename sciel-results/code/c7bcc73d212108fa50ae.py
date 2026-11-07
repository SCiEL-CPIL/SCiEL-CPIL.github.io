import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    colors = [-1] * n
    for i in range(n):
        if colors[i] == -1:
            colors[i] = 0
            q = [i]
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        return False
    return True

def _valid_solution(n, base_edges, added_edges):
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    diam = _bfs_diameter(n, adj)
    return diam != -1 and diam <= 4

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # Generate all valid candidate edges (opposite parity, not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                candidates.append((u, v))
    
    deadline = time.time() + 80
    
    # Try k from 3 to 7
    for k in range(3, 8):
        if time.time() > deadline:
            break
            
        # Try several random restarts for this k
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Greedy construction
            added = []
            current_edges = list(base_edges)
            
            # We want to add exactly k edges if possible, or fewer if diameter becomes <= 4
            # But we are iterating k, so we target k edges.
            # Actually, if we find a solution with < k edges in a previous iteration, we would have returned.
            # So here we assume we need at least k edges.
            
            # Greedy: Add k edges that best reduce diameter
            # To allow randomness, we can shuffle candidates or pick randomly from top few
            adj = _build_adj(n, current_edges)
            
            for i in range(k):
                if time.time() > deadline:
                    break
                
                # Find current diameter
                diam = _bfs_diameter(n, adj)
                if diam <= 4:
                    # We achieved diameter 4 with fewer than k edges?
                    # This shouldn't happen if we checked smaller k first, 
                    # but maybe the greedy path led here.
                    # Return the current added edges if valid.
                    if _valid_solution(n, base_edges, added):
                        return [[u, v] for u, v in added]
                    
                    # If valid but we wanted k, we can stop here.
                    # But we are in loop for k, so smaller k would have caught it.
                    # Unless the first k-1 didn't work but k-1 subset of this does?
                    # Unlikely with greedy.
                    
                # Find best candidate edge to add
                best_edge = None
                best_diam = diam if diam != -1 else 100
                added_set = set((min(a,b), max(a,b)) for a,b in added)
                
                # To add randomness, we can sort candidates by potential benefit
                # But calculating benefit for all is expensive.
                # Instead, pick a random subset of candidates to evaluate, or evaluate all.
                # With n=24, candidates ~ 120. Evaluating 120 BFS is fast.
                
                # Shuffle candidates to break ties / add randomness
                local_candidates = candidates[:]
                random.shuffle(local_candidates)
                
                for a, b in local_candidates:
                    if (min(a,b), max(a,b)) in added_set:
                        continue
                    
                    # Add edge
                    adj[a].add(b)
                    adj[b].add(a)
                    
                    new_diam = _bfs_diameter(n, adj)
                    
                    # Remove edge
                    adj[a].remove(b)
                    adj[b].remove(a)
                    
                    if new_diam != -1 and new_diam < best_diam:
                        best_diam = new_diam
                        best_edge = (a, b)
                        
                        # If we reached target diameter, we can stop early?
                        # No, we must add exactly k edges to satisfy the loop logic?
                        # No, the goal is to MINIMIZE edges.
                        # If we reach diameter 4 with i < k edges, we should return.
                        if new_diam <= 4:
                            # Check validity
                            added.append((a, b))
                            if _valid_solution(n, base_edges, added):
                                return [[u, v] for u, v in added]
                            # If invalid, continue? 
                            # Triangle free is guaranteed by parity.
                            # Connected is guaranteed by path.
                            # So if diam <= 4, it is valid.
                            # So we return.
                            pass 

                if best_edge is None:
                    break
                    
                added.append(best_edge)
                current_edges.append(best_edge)
                adj[best_edge[0]].add(best_edge[1])
                adj[best_edge[1]].add(best_edge[0])
                
            # After adding k edges, check if valid
            if _valid_solution(n, base_edges, added):
                return [[u, v] for u, v in added]
                
    # Fallback: return empty
    return []

# EVOLVE-BLOCK-END
