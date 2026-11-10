import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Vertices 0..14
    # Base graph: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
    
    # Precompute all possible intra-part edges
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append((B[i], B[j]))
    
    # Build adjacency and degree tracking for fast validation
    def build_adj(deg, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_valid_fast(deg, adj, edges):
        """Check if no path s-u-v-t has deg[s]==deg[t]"""
        # Optimization: Precompute neighbors
        for u, v in edges:
            # Neighbors of u excluding v
            nu = adj[u]
            nv = adj[v]
            for s in nu:
                if s == v:
                    continue
                for t in nv:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def is_valid_edges(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, edges)
        return check_valid_fast(deg, adj, edges)
    
    # Greedy with multiple orderings
    def greedy_add(base_edge_set, cand_list):
        curr = set(base_edge_set)
        deg = [0] * n
        for u, v in curr:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, curr)
        
        for c in cand_list:
            if c in curr:
                continue
            u, v = c
            
            # Add edge to structures
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
            if check_valid_fast(deg, adj, curr | {c}):
                curr.add(c)
            else:
                # Revert
                deg[u] -= 1
                deg[v] -= 1
                adj[u].remove(v)
                adj[v].remove(u)
        
        return curr
    
    # Local search: try swapping edges or adding edges
    def local_search(edge_set, max_iters=50):
        curr = set(edge_set)
        deg = [0] * n
        for u, v in curr:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, curr)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding any candidate
            for c in candidates:
                if c in curr:
                    continue
                u, v = c
                
                # Add edge
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
                if check_valid_fast(deg, adj, curr | {c}):
                    curr.add(c)
                    improved = True
                    break
                else:
                    # Revert
                    deg[u] -= 1
                    deg[v] -= 1
                    adj[u].remove(v)
                    adj[v].remove(u)
            
            if not improved:
                # Try removing an edge and adding two others? 
                # Or just try removing one edge and see if we can add more later?
                # For now, just try to add. If stuck, try a swap: remove one, add one.
                # If we remove e1 and add e2, size stays same.
                # If we remove e1 and add e2, e3, size increases.
                
                # Let's try a simple swap first to escape local minima
                found_swap = False
                for e1 in list(curr):
                    u1, v1 = e1
                    # Remove e1
                    deg[u1] -= 1
                    deg[v1] -= 1
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    curr.remove(e1)
                    
                    # Try to add any candidate
                    for c in candidates:
                        if c in curr:
                            continue
                        u, v = c
                        
                        # Add c
                        deg[u] += 1
                        deg[v] += 1
                        adj[u].add(v)
                        adj[v].add(u)
                        
                        if check_valid_fast(deg, adj, curr | {c}):
                            curr.add(c)
                            found_swap = True
                            improved = True
                            break
                        else:
                            # Revert c
                            deg[u] -= 1
                            deg[v] -= 1
                            adj[u].remove(v)
                            adj[v].remove(u)
                    
                    if found_swap:
                        break
                    
                    # Revert e1
                    curr.add(e1)
                    deg[u1] += 1
                    deg[v1] += 1
                    adj[u1].add(v1)
                    adj[v1].add(u1)
        
        return curr
    
    # Initial best solution
    best_edges = set(base_edges)
    best_count = len(best_edges)
    
    # Multiple greedy passes with different orderings
    for restart in range(200):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # First pass: sort by minimum degree endpoints
            ordered = sorted(candidates, key=lambda c: (min(c[0], c[1]), max(c[0], c[1])))
        elif restart % 3 == 1:
            ordered = candidates[:]
            random.shuffle(ordered)
        else:
            # Sort by "centrality" - prefer edges between high-degree potential vertices
            ordered = candidates[:]
            random.shuffle(ordered)
        
        curr = greedy_add(base_edges, ordered)
        
        if len(curr) > best_count:
            best_edges = curr
            best_count = len(curr)
        
        # Local search on the best result
        if best_count >= 56:
            if time.time() > deadline:
                break
            improved = local_search(best_edges, max_iters=50)
            if len(improved) > best_count:
                best_edges = improved
                best_count = len(best_edges)
    
    result = [list(e) for e in best_edges]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result
