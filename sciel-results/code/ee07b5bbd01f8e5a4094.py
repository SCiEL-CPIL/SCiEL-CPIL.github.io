import random
import time
from itertools import combinations

def construct():
    n = 18
    # Target: Minimize max triangle-free induced subgraph size
    # Constraint: K4-free
    
    def is_k4_free(edges_set, u, v):
        # Check if adding (u,v) creates a K4
        # Neighbors of u and v must have no common neighbor
        # that is connected to both u and v?
        # Actually, a K4 is formed if there exist x, y such that
        # (u,x), (u,y), (v,x), (v,y), (x,y) are all edges.
        # Simpler: For every pair x, y in V \ {u, v}:
        # if (u,x) in E and (v,x) in E and (u,y) in E and (v,y) in E and (x,y) in E:
        #   then adding (u,v) creates K4 {u,v,x,y}.
        
        # Efficient check:
        # Common neighbors of u and v in current graph?
        # No, we are checking if adding (u,v) creates a K4.
        # The K4 would consist of u, v, and two other vertices x, y.
        # x and y must be adjacent to u, adjacent to v, and adjacent to each other.
        # So, let N_u = neighbors of u, N_v = neighbors of v.
        # We need to find x, y in N_u ∩ N_v such that (x,y) is an edge.
        # If such x, y exist, adding (u,v) creates a K4.
        
        n_u = [x for x in range(n) if (min(u,x), max(u,x)) in edges_set]
        n_v = [x for x in range(n) if (min(v,x), max(v,x)) in edges_set]
        common = set(n_u).intersection(n_v)
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                x = common_list[i]
                y = common_list[j]
                if (min(x,y), max(x,y)) in edges_set:
                    return False
        return True

    def max_triangle_free_size(edges_set):
        # Exact branch and bound for max subset with no triangles
        # n=18, so we can use a recursive search
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) not in edges_set: continue
                for w in range(v+1, n):
                    if (v,w) in edges_set and (u,w) in edges_set:
                        triangles.append((u,v,w))
        
        # If no triangles, return n
        if not triangles:
            return n
            
        # Map vertex to list of triangles it participates in
        # For pruning: if we include u, we must exclude all pairs (v,w) that form a triangle with u.
        # This is hard to represent as simple vertex exclusion.
        
        # Alternative: Iterate all subsets? 2^18 = 262k.
        # Check if subset is triangle-free.
        # A subset S is triangle-free iff it does not contain any triangle.
        # We can check this by iterating through the list of triangles.
        # If any triangle is fully contained in S, S is invalid.
        
        best = 0
        # Use a greedy heuristic to get a good lower bound for pruning?
        # Or just brute force with bitmasks.
        
        # Bitmask representation of vertices
        # For each triangle, create a bitmask
        tri_masks = []
        for t in triangles:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            tri_masks.append(m)
            
        # Iterate all masks from 1 to 2^n - 1
        # Optimization: Iterate by size descending? No, just check all.
        # 262,144 iterations.
        # For each mask, check if it contains any tri_mask.
        # This is fast in Python if we use integers.
        
        # Actually, we can just check if (mask & tri_mask) == tri_mask for any tri_mask.
        # If yes, it's not triangle-free.
        
        # To speed up, we can sort triangles or use a set?
        # Just a loop over triangles is fine.
        # Max triangles ~ 800.
        # 262k * 800 = 200 million ops. Might be ~1-2 seconds in Python.
        # We need this to be faster for local search.
        
        # Better: Use the property that if a set is triangle-free, its size is limited.
        # We want the MAX size.
        # We can iterate sizes from n down to 1?
        # No, we can just find the max.
        
        # Let's try a recursive branch and bound on vertices.
        # Sort vertices by degree?
        
        def dfs(available_mask, current_size):
            nonlocal best
            if current_size + bin(available_mask).count('1') <= best:
                return
            if available_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex, say the lowest set bit
            v = (available_mask & -available_mask).bit_length() - 1
            # Option 1: Exclude v
            dfs(available_mask & ~(1 << v), current_size)
            
            # Option 2: Include v
            # If we include v, we cannot include any pair (u,w) that forms a triangle with v.
            # This means for any u in available, if (u, w) is an edge and (v, u) is edge and (v, w) is edge...
            # Wait, the condition is that the *set* has no triangle.
            # If we include v, we must ensure no other two vertices in the set form a triangle with v.
            # So, for every u in available \ {v}, if (v,u) is an edge, then we cannot include any w such that (v,w) is edge and (u,w) is edge.
            # This effectively removes many vertices.
            
            # Let's identify vertices that cannot be included together with v.
            # If we include v, and we include u, then we cannot include any w such that {v,u,w} is a triangle.
            # This is getting complicated to mask out dynamically.
            
            # Simpler: Just check all subsets of size > best?
            # Or use the bitmask loop but optimized.
            
            pass # This DFS is hard to optimize quickly.
            
        # Let's stick to the bitmask loop but make it faster.
        # We only need to check masks of size > best.
        # Initialize best with a greedy independent set size (which is a lower bound for triangle-free? No, independent set is triangle-free).
        # Greedy: pick vertices with lowest degree in remaining graph?
        
        # Let's just do the full loop but break early if possible?
        # Actually, for n=18, the loop is acceptable if we do it a limited number of times.
        
        # Optimization: Precompute for each vertex v, the list of triangles containing v.
        # If a mask contains v, and contains u, and contains w, and {v,u,w} is a triangle, it's invalid.
        
        # Let's use a different approach:
        # Iterate over all masks.
        # If mask is triangle-free, update best.
        
        # To speed up:
        # Only check masks that are supersets of some "core"? No.
        
        # Let's try the loop.
        for mask in range(1, 1 << n):
            # Quick check: if bin(mask).count('1') <= best: continue
            # This check is expensive.
            
            # Check if mask contains any triangle
            valid = True
            for tm in tri_masks:
                if (mask & tm) == tm:
                    valid = False
                    break
            if valid:
                size = bin(mask).count('1')
                if size > best:
                    best = size
                    # If best reaches n, we can stop? No, we want max.
                    # If best is n, return n.
                    if best == n:
                        return n
                        
        return best

    # Initialization: Random 6-regular graph
    # Use configuration model or random edges until regular
    def generate_random_regular_graph(n, k):
        # Simple: random edges, check degrees
        # For n=18, k=6
        while True:
            edges = set()
            # Try to build a k-regular graph
            # Random permutation of half-edges
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            valid = True
            for i in range(0, n*k, 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            if valid and len(edges) == n*k//2:
                return edges

    edges = generate_random_regular_graph(n, 6)
    # Ensure K4-free? Random 6-regular on 18 vertices might have K4s.
    # We need to remove edges that create K4s.
    # Check all K4s and remove one edge from each.
    changed = True
    while changed:
        changed = False
        # Find all K4s
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) not in edges: continue
                for x in range(v+1, n):
                    if (u,x) not in edges or (v,x) not in edges: continue
                    for y in range(x+1, n):
                        if (u,y) in edges and (v,y) in edges and (x,y) in edges:
                            # K4 found: u,v,x,y
                            # Remove a random edge from this K4
                            k4_edges = [(u,v), (u,x), (u,y), (v,x), (v,y), (x,y)]
                            rem_edge = random.choice(k4_edges)
                            edges.remove(rem_edge)
                            changed = True
                            break
                    if changed: break
                if changed: break
            if changed: break

    current_score = max_triangle_free_size(edges)
    
    # Local Search
    # Try to improve by flipping edges
    # Time limit check
    start_time = time.time()
    limit = 100 # seconds
    
    while time.time() - start_time < limit:
        improved = False
        # Try random edge flips
        for _ in range(100):
            if time.time() - start_time > limit: break
            # Pick a random pair
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            u, v = min(u,v), max(u,v)
            
            if (u,v) in edges:
                # Remove edge
                new_edges = edges.copy()
                new_edges.remove((u,v))
                new_score = max_triangle_free_size(new_edges)
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    improved = True
                    break
            else:
                # Add edge
                if is_k4_free(edges, u, v):
                    new_edges = edges.copy()
                    new_edges.add((u,v))
                    new_score = max_triangle_free_size(new_edges)
                    if new_score < current_score:
                        edges = new_edges
                        current_score = new_score
                        improved = True
                        break
        if not improved:
            # Restart with new random graph
            edges = generate_random_regular_graph(n, 6)
            # Remove K4s again
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in range(u+1, n):
                        if (u,v) not in edges: continue
                        for x in range(v+1, n):
                            if (u,x) not in edges or (v,x) not in edges: continue
                            for y in range(x+1, n):
                                if (u,y) in edges and (v,y) in edges and (x,y) in edges:
                                    k4_edges = [(u,v), (u,x), (u,y), (v,x), (v,y), (x,y)]
                                    rem_edge = random.choice(k4_edges)
                                    edges.remove(rem_edge)
                                    changed = True
                                    break
                            if changed: break
                        if changed: break
                    if changed: break
            current_score = max_triangle_free_size(edges)

    return [list(e) for e in edges]
