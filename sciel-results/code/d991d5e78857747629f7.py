import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def is_sum_free(n, S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if (a + b) % n in S_set:
                    return False
        return True

    def build_adj(n, S):
        adj = [0] * n
        for d in S:
            for i in range(n):
                j = (i + d) % n
                adj[i] |= (1 << j)
        return adj

    def has_independent_set_10(adj, n):
        # Returns True if there is an independent set of size 10
        # We check if the complement graph has a clique of size 10
        # Complement adjacency: ~adj[i] & ((1<<n)-1)
        # But we can just do a DFS for independent set in G directly.
        # G is triangle-free, so it's sparse-ish.
        # Actually, checking for independent set of size 10 in G is the goal.
        
        # Use a simple backtracking with bitsets
        # candidates is a bitmask of vertices that can be added to the current independent set
        # current_set_size is the size of the current independent set
        
        def dfs(candidates, size):
            if size >= 10:
                return True
            if not candidates:
                return False
            
            # Heuristic: pick a vertex with low degree in the candidate subgraph?
            # Or just pick the first one.
            # To be faster, we can use a bound: if size + popcount(candidates) < 10, return False
            
            # Popcount for Python int
            if size + bin(candidates).count('1') < 10:
                return False

            # Pick a vertex v from candidates
            # Let's pick the one with the smallest degree in the candidate set to prune better?
            # For simplicity and speed, pick the lowest set bit.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates are those in candidates that are NOT adjacent to v
            new_cands = candidates & ~adj[v]
            if dfs(new_cands, size + 1):
                return True
            
            # Branch 2: Exclude v
            new_cands2 = candidates & ~v_bit
            if dfs(new_cands2, size):
                return True
                
            return False

        all_mask = (1 << n) - 1
        return dfs(all_mask, 0)

    def find_circulant(n):
        # Try to find a sum-free set S for n such that alpha < 10
        # Max possible size of S is roughly n/3
        # We want S to be as large as possible to make G dense, thus alpha small
        
        # Start with large sets and shrink? Or random search?
        # Random search over subsets of {1, ..., n//2}
        
        max_diff = n // 2
        candidates_diffs = list(range(1, max_diff + 1))
        
        best_S = None
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            # Random subset
            k = random.randint(max_diff // 3, max_diff // 2)
            S = set(random.sample(candidates_diffs, k))
            
            if not is_sum_free(n, S):
                continue
                
            adj = build_adj(n, S)
            
            # Check independent set
            if not has_independent_set_10(adj, n):
                # Found a valid graph
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return edges

        return None

    # Try n from 40 down to 30
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        edges = find_circulant(n)
        if edges is not None:
            return {'n': n, 'edges': edges}

    # Fallback: Random Local Search for n=40
    # Initialize with random edges, remove triangles, break independent sets
    
    n = 40
    adj = [0] * n
    all_mask = (1 << n) - 1
    
    # Start with a random graph
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Remove triangles
    # A triangle is 0-1-2. If 0-1, 1-2, 0-2 exist, remove one.
    # We can iterate and remove.
    for _ in range(1000):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                # Check common neighbors
                common = adj[i] & adj[j]
                # If common has any bit set, there is a triangle
                if common:
                    # Remove edge (i, j)
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    changed = True
        if not changed:
            break

    # Now G is triangle-free.
    # Check for independent set of size 10.
    # If exists, add edges to break it.
    
    def get_independent_set(adj, n, size):
        # Returns a set of vertices forming an independent set of given size, or None
        def dfs(candidates, current_set, size_needed):
            if size_needed == 0:
                return current_set
            if not candidates or bin(candidates).count('1') < size_needed:
                return None
            
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Include v
            new_cands = candidates & ~adj[v]
            res = dfs(new_cands, current_set | {v}, size_needed - 1)
            if res:
                return res
            
            # Exclude v
            res = dfs(candidates & ~v_bit, current_set, size_needed)
            if res:
                return res
            return None
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, set(), size)

    for _ in range(1000):
        if time.time() > deadline:
            break
            
        ind_set = get_independent_set(adj, n, 10)
        if not ind_set:
            # Valid graph found
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Break the independent set by adding an edge
        # Pick two vertices from ind_set that are not adjacent (they aren't, by definition)
        # Adding an edge might create a triangle.
        # We must ensure no triangle is created.
        # Let u, v be the vertices.
        # Triangle is created if they have a common neighbor.
        # Since they are in an independent set, they have no common neighbors *within the set*,
        # but they might have common neighbors *outside*.
        # If they have a common neighbor w, adding (u,v) creates triangle (u,v,w).
        # So we can only add (u,v) if they have NO common neighbors.
        
        found_edge = False
        verts = list(ind_set)
        random.shuffle(verts)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u = verts[i]
                v = verts[j]
                # Check if adding (u,v) creates a triangle
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                if not common:
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    found_edge = True
                    break
            if found_edge:
                break
        
        if not found_edge:
            # Could not break this independent set without creating triangles
            # Try to restart or modify graph differently
            # For now, just break and try again with new random init
            break

    # If all else fails, return a small valid graph
    # n=26 fallback from seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
