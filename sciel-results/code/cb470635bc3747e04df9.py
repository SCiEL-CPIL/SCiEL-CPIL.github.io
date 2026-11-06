import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        """Return list of bitmasks for all triangles in the graph."""
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size(adj):
        """
        Find the largest induced triangle-free subgraph using branch and bound.
        Optimized for speed.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Precompute triangle masks per vertex for fast check
        # tri_mask[v] is a bitmask of triangle indices containing v
        tri_count = len(triangles)
        tri_mask = [0] * n
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_mask[v] |= (1 << idx)
        
        best_size = 0
        
        # Order vertices by degree in triangle graph?
        # Or just use bit order.
        
        def backtrack(current_mask, remaining_mask, current_size):
            nonlocal best_size
            
            # Pruning
            if current_size + bin(remaining_mask).count('1') <= best_size:
                return
            
            if remaining_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick vertex with lowest index in remaining for determinism
            # Or pick vertex with most triangles in remaining?
            # Let's pick the one with most constraints to prune faster
            # But calculating that is expensive. Let's stick to lowest index.
            v = (remaining_mask & (-remaining_mask)).bit_length() - 1
            remaining_without_v = remaining_mask & ~(1 << v)
            
            # Option 1: Exclude v
            backtrack(current_mask, remaining_without_v, current_size)
            
            # Option 2: Include v
            # Check if v can be added
            # A triangle is formed if there exists a triangle t such that 
            # t is subset of (current_mask | (1<<v))
            # Since t has 3 vertices, and v is one, the other 2 must be in current_mask.
            # So check if any triangle containing v has its other 2 vertices in current_mask.
            
            can_add = True
            # Iterate through triangles containing v
            # tri_mask[v] has bits set for triangles containing v
            tm = tri_mask[v]
            # We need to check each triangle index in tm
            # But iterating through all triangles is slow if many.
            # Better: For each triangle t, check if (t & ~(1<<v)) is subset of current_mask.
            # We can precompute for each v, the list of "other two" masks for triangles containing v.
            # Let's do that outside.
            
            if can_add:
                backtrack(current_mask | (1 << v), remaining_without_v, current_size + 1)
        
        # Precompute "other two" masks for each vertex
        # other_two[v][i] is the mask of the other 2 vertices in triangle i (if v in i)
        # Or just a list of masks for each v.
        other_two_masks = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    other_two_masks[v].append(t & ~(1 << v))
        
        def backtrack_optimized(current_mask, remaining_mask, current_size):
            nonlocal best_size
            
            if current_size + bin(remaining_mask).count('1') <= best_size:
                return
            
            if remaining_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            v = (remaining_mask & (-remaining_mask)).bit_length() - 1
            remaining_without_v = remaining_mask & ~(1 << v)
            
            # Exclude v
            backtrack_optimized(current_mask, remaining_without_v, current_size)
            
            # Include v
            can_add = True
            for mask in other_two_masks[v]:
                if (current_mask & mask) == mask:
                    can_add = False
                    break
            
            if can_add:
                backtrack_optimized(current_mask | (1 << v), remaining_without_v, current_size + 1)
        
        backtrack_optimized(0, (1 << n) - 1, 0)
        return best_size

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(200):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            # Remove random edge from K4 {a,b,c,d}
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    
    candidates = []
    
    # 1. K_{6,6,6} - Strong candidate for min triangle-free size (12)
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)
    
    # 2. Other Turan graphs? T(18,4) is K5-free, but we need K4-free.
    # T(18,3) is the only balanced Turan for K4-free.
    
    # 3. Random dense graphs (might find better structures)
    for _ in range(50):
        if time.time() > deadline - 40:
            break
        density = 0.6 + random.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)
            
    # Evaluate candidates
    best_adj = None
    best_score = 18
    
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj)
        
    # Local Search: Try to improve from best_adj
    adj = best_adj[:]
    current_score = best_score
    
    # Simulated Annealing
    temperature = 1.0
    cooling = 0.995
    no_improve_count = 0
    
    for iteration in range(3000):
        if time.time() > deadline - 10:
            break
        if best_score <= 10: # Theoretical lower bound might be lower, but 10 is good
            break
            
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        
        # Flip edge
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        new_score = max_triangle_free_size(adj)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            adj = old_adj
            no_improve_count += 1
        
        temperature *= cooling
        
        if no_improve_count > 300:
            no_improve_count = 0
            temperature = max(temperature, 1.0)
            
    return edges_from_adj(best_adj)
