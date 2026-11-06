import random
import time

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_degrees(adj, n):
    return [len(adj[i]) for i in range(n)]

def _find_triangles(adj, n):
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    triangles.append((i, j, k))
    return triangles

def _max_triangle_degree_sum(triangles, degrees):
    if not triangles:
        return 0
    return max(degrees[u] + degrees[v] + degrees[w] for u, v, w in triangles)

def construct():
    n = 16
    # Start with K_{8,8} plus one edge inside part A
    part_a = list(range(8))
    part_b = list(range(8, 16))
    
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
    edges.append([0, 1])
    
    adj = _build_adjacency(n, edges)
    degrees = _compute_degrees(adj, n)
    
    best_score = float('inf')
    best_adj = [set(s) for s in adj]
    
    # We will run multiple local searches from different starts or restarts
    # Given time limit, we should do a very fast local search.
    
    start_time = time.time()
    max_time = 110 # seconds
    
    # Helper to try a move
    def try_move(adj, degrees, triangles, max_sum):
        if not triangles:
            return False, adj, degrees, triangles, max_sum
        
        # Find worst triangle
        worst_tri = None
        worst_sum = -1
        for t in triangles:
            s = degrees[t[0]] + degrees[t[1]] + degrees[t[2]]
            if s > worst_sum:
                worst_sum = s
                worst_tri = t
        
        if worst_sum < best_score:
            # Update best if we find something better globally? 
            # No, best_score is for the final return. 
            # We want to minimize the current state's max sum.
            pass
            
        u, v, w = worst_tri
        tri_edges = [(u, v), (v, w), (u, w)]
        
        # Sort edges by degree sum of endpoints to prioritize removing high-degree edges
        tri_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
        
        for a, b in tri_edges:
            if b not in adj[a]:
                continue
                
            # Candidate non-edges: prioritize low degree vertices
            # Generate list of non-edges sorted by degree sum
            non_edges = []
            for c in range(n):
                for d in range(c + 1, n):
                    if d not in adj[c]:
                        non_edges.append((c, d))
            
            # Sort by degrees[c] + degrees[d]
            non_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]])
            
            # Try top K candidates
            for c, d in non_edges[:50]:
                # Check if adding (c,d) creates a triangle with sum > max_sum
                # New degrees:
                # deg(a)-1, deg(b)-1, deg(c)+1, deg(d)+1
                
                # Common neighbors of c and d
                common = adj[c] & adj[d]
                
                # Check new triangles formed by (c,d,x) for x in common
                valid = True
                for x in common:
                    new_sum = (degrees[c] + 1) + (degrees[d] + 1) + degrees[x]
                    # Note: if x is a or b, degrees change too.
                    # This approximation is risky. Let's do a more precise check for the worst case.
                    # Actually, let's just simulate the swap for the top few candidates.
                    pass
                
                # Precise check for this specific swap
                # We need to check if ANY triangle in the new graph has sum > max_sum.
                # This is expensive. 
                # Optimization: Only check triangles that are affected.
                # Affected triangles:
                # 1. Triangles containing a or b (might lose an edge or change degree)
                # 2. Triangles containing c or d (might gain an edge or change degree)
                # 3. New triangles (c, d, x)
                
                # Let's just do a full re-evaluation for the top 5 candidates if they look promising.
                # Or, we can use a heuristic: 
                # If degrees[c] + degrees[d] + max(deg[x] for x in common) <= max_sum, it's likely safe.
                
                if common:
                    max_common_deg = max(degrees[x] for x in common)
                    if (degrees[c] + 1) + (degrees[d] + 1) + max_common_deg > max_sum:
                        continue
                
                # Also check if removing (a,b) breaks a triangle that was the *only* thing keeping the sum high?
                # No, removing an edge reduces degrees, which is good.
                
                # Let's verify by actually performing the swap in a copy? Too slow.
                # Let's assume if the new triangles are okay, and we reduce degrees of a,b, 
                # the existing triangles involving a or b will have lower sums.
                # Existing triangles involving c or d will have higher sums.
                
                # Check existing triangles involving c or d
                for t in triangles:
                    if c in t or d in t:
                        # Calculate new sum
                        x1, x2, x3 = t
                        new_sum = 0
                        for x in t:
                            if x == a: new_sum += degrees[a] - 1
                            elif x == b: new_sum += degrees[b] - 1
                            elif x == c: new_sum += degrees[c] + 1
                            elif x == d: new_sum += degrees[d] + 1
                            else: new_sum += degrees[x]
                        
                        if new_sum > max_sum:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Perform swap
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].add(d)
                adj[d].add(c)
                
                # Update degrees
                degrees[a] -= 1
                degrees[b] -= 1
                degrees[c] += 1
                degrees[d] += 1
                
                # Update triangles
                # Remove triangles containing (a,b)
                new_triangles = []
                for t in triangles:
                    if (a in t and b in t):
                        continue
                    new_triangles.append(t)
                
                # Add new triangles (c,d,x)
                for x in common:
                    new_triangles.append((c, d, x))
                
                # Re-sort or just keep list
                triangles = new_triangles
                max_sum = _max_triangle_degree_sum(triangles, degrees)
                
                return True, adj, degrees, triangles, max_sum
        
        return False, adj, degrees, triangles, max_sum

    # Main loop
    adj = _build_adjacency(n, edges)
    degrees = _compute_degrees(adj, n)
    triangles = _find_triangles(adj, n)
    max_sum = _max_triangle_degree_sum(triangles, degrees)
    
    best_global_score = max_sum
    best_global_adj = [set(s) for s in adj]
    
    for _ in range(50): # 50 restarts
        if time.time() - start_time > max_time:
            break
            
        # Randomize start slightly? Or just start from current best?
        # Let's start from the current best and try to improve.
        # If stuck, do a random perturbation.
        
        current_adj = [set(s) for s in best_global_adj]
        current_degrees = _compute_degrees(current_adj, n)
        current_triangles = _find_triangles(current_adj, n)
        current_max = _max_triangle_degree_sum(current_triangles, current_degrees)
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved, current_adj, current_degrees, current_triangles, current_max = try_move(
                current_adj, current_degrees, current_triangles, current_max
            )
            
            if current_max < best_global_score:
                best_global_score = current_max
                best_global_adj = [set(s) for s in current_adj]
                
                if best_global_score < 24:
                    # We beat the published construction!
                    # We can break early if we hit a very low score?
                    # The conjecture is ~23.4.
                    if best_global_score <= 23:
                        break

        # If stuck, perturb
        if time.time() - start_time > max_time:
            break
            
        # Perturb best_global_adj
        if random.random() < 0.5:
            # Random swap
            edge_list = [[min(u, v), max(u, v)] for u in range(n) for v in best_global_adj[u] if u < v]
            if edge_list:
                idx = random.randint(0, len(edge_list) - 1)
                a, b = edge_list[idx]
                non_edges = []
                for c in range(n):
                    for d in range(c + 1, n):
                        if d not in best_global_adj[c]:
                            non_edges.append((c, d))
                if non_edges:
                    c, d = random.choice(non_edges)
                    best_global_adj[a].remove(b)
                    best_global_adj[b].remove(a)
                    best_global_adj[c].add(d)
                    best_global_adj[d].add(c)
                    
    # Reconstruct edges from best_global_adj
    edges = []
    for u in range(n):
        for v in best_global_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
