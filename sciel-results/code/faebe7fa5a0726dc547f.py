import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def has_independent_set_10(adj, n):
        """
        Check if graph has independent set of size 10.
        Uses bitset branch and bound.
        Returns True if it does.
        """
        mask_all = (1 << n) - 1
        
        # Pre-compute bit counts for small masks if needed, but bit_count() is fast in Py3.10+
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Heuristic: pick vertex with max degree in candidates
            # This reduces the candidate set the most in the 'include' branch
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            # New candidates: candidates minus best_v and its neighbors
            new_cand_include = candidates & ~(1 << best_v) & ~adj[best_v]
            if backtrack(new_cand_include, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            # New candidates: candidates minus best_v
            new_cand_exclude = candidates & ~(1 << best_v)
            if backtrack(new_cand_exclude, size):
                return True
                
            return False
        
        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp ^= lsb
        return edges

    def make_andrasfai(k):
        """Andrásfai graph And(k): vertices Z_{3k}, i~j iff (j-i) mod 3k in {1, 3k-1}.
        Triangle-free, alpha = k."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def random_triangle_free(n, p):
        """
        Generate a random triangle-free graph.
        We iterate through all possible edges in random order.
        If adding the edge doesn't create a triangle, we add it with probability p.
        """
        adj = [0] * n
        # Generate list of all edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates a triangle
            # A triangle (u,v,w) exists if there is a common neighbor w of u and v
            common = adj[u] & adj[v]
            if common == 0:
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def local_search_for_alpha(adj, n, max_iter=1000):
        """
        Try to reduce the independence number by adding edges.
        Returns the modified adj if alpha < 10 is achieved, or None.
        This is a heuristic.
        """
        # We need to be careful not to create triangles.
        # We don't strictly need to minimize alpha, just get it < 10.
        # If current alpha < 10, we are done.
        if not has_independent_set_10(adj, n):
            return adj
        
        # Find a maximum independent set (or a large one)
        # If we can't find one of size 10, we are done.
        # The check above already does this.
        # So this function is only called if alpha >= 10.
        
        # To reduce alpha, we should add edges within the independent set.
        # However, we don't have the set itself from the boolean check.
        # Let's find one.
        
        def find_independent_set(adj, n, target_size):
            mask_all = (1 << n) - 1
            found = []
            
            def backtrack(candidates, current_set):
                if len(current_set) == target_size:
                    return True
                if candidates == 0:
                    return False
                if len(current_set) + candidates.bit_count() < target_size:
                    return False
                
                # Pick vertex with min degree in candidates to keep options open?
                # Or max degree to prune? For finding a set, min degree is often better.
                best_v = -1
                min_deg = n + 1
                temp = candidates
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = (adj[v] & candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                
                if best_v == -1:
                    return False
                
                # Include
                current_set.append(best_v)
                new_cand = candidates & ~(1 << best_v) & ~adj[best_v]
                if backtrack(new_cand, current_set):
                    return True
                current_set.pop()
                
                # Exclude
                new_cand = candidates & ~(1 << best_v)
                if backtrack(new_cand, current_set):
                    return True
                
                return False
            
            if backtrack(mask_all, found):
                return found
            return None

        # This local search is complex and slow. 
        # Better strategy: Just try many random graphs.
        # Random triangle-free graphs with high p tend to have small alpha.
        return None

    best_result = None
    
    # 1. Try known constructions
    # Andrásfai And(9) has n=27, alpha=9. Valid.
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
        
    # Try And(10) is n=30, alpha=10. Invalid.
    # But maybe a modified version?
    
    # 2. Random Search for n=40, 39, 38...
    # We want to maximize n.
    # Start from 40 down to best_result['n'] + 1
    
    min_target = (best_result['n'] + 1) if best_result else 28
    
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 20000 # Adjust based on speed
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random triangle-free graph
            # p should be high enough to keep alpha low.
            # For n=40, alpha < 10 is required.
            # Turan graph T(40, 2) is bipartite, alpha=20. Too high.
            # We need a graph that is not bipartite but triangle-free.
            # Andrásfai graphs are good.
            
            # Strategy: Start with Andrásfai And(9) (n=27) and add vertices?
            # Or just random triangle-free.
            
            # Let's try random triangle-free with varying p
            p = random.uniform(0.4, 0.8)
            adj = random_triangle_free(target_n, p)
            
            # Check validity
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
                
            # If we fail, maybe try a different strategy
            # e.g. Blow up And(3) or And(4)
            if attempts % 10 == 0:
                 # Blow up And(3) n=9
                 # Random sizes summing to target_n
                 sizes = [1] * 9
                 rem = target_n - 9
                 for _ in range(rem):
                     sizes[random.randint(0, 8)] += 1
                 
                 # Construct blow up
                 base_adj, _ = make_andrasfai(3)
                 offsets = [0] * 10
                 for i in range(9):
                     offsets[i+1] = offsets[i] + sizes[i]
                 
                 new_adj = [0] * target_n
                 for i in range(9):
                     for j in range(9):
                         if i != j and (base_adj[i] & (1 << j)):
                             for a in range(offsets[i], offsets[i+1]):
                                 for b in range(offsets[j], offsets[j+1]):
                                     new_adj[a] |= (1 << b)
                                     new_adj[b] |= (1 << a)
                                     
                 if not has_independent_set_10(new_adj, target_n):
                     found = True
                     best_result = {'n': target_n, 'edges': build_edge_list(new_adj, target_n)}
                     break
                     
        if found:
            break
            
    if best_result is None:
        # Fallback
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
        
    # Final verification
    final_adj = [0] * best_result['n']
    for u, v in best_result['edges']:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
        
    assert is_triangle_free(final_adj, best_result['n']), "Result is not triangle-free!"
    assert not has_independent_set_10(final_adj, best_result['n']), "Result has independent set of size 10!"
    
    return best_result
