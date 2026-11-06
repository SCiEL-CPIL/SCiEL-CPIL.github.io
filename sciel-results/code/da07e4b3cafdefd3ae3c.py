import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = [j for j in range(n) if adj[i] & (1 << j)]
            for u in range(len(neighbors)):
                for v in range(u+1, len(neighbors)):
                    if adj[neighbors[u]] & (1 << neighbors[v]):
                        return False
        return True

    def max_independent_set_size(adj, n):
        best = 0
        def dfs(remaining_mask, current_size):
            nonlocal best
            if current_size >= 10:
                return 10
            if remaining_mask == 0:
                if current_size > best:
                    best = current_size
                return best
            if current_size + bin(remaining_mask).count('1') <= best:
                return best
            
            # Pick vertex with max degree in remaining
            max_deg = -1
            chosen = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
                temp_mask ^= lsb
            
            if chosen == -1:
                return best

            # Include
            new_mask = remaining_mask & ~(adj[chosen] | (1 << chosen))
            res1 = dfs(new_mask, current_size + 1)
            if res1 >= 10: return 10
            
            # Exclude
            new_mask = remaining_mask & ~(1 << chosen)
            res2 = dfs(new_mask, current_size)
            if res2 >= 10: return 10
            
            return max(res1, res2)

        return dfs((1 << n) - 1, 0)

    def is_valid(adj, n):
        if not is_triangle_free(adj, n):
            return False
        if max_independent_set_size(adj, n) >= 10:
            return False
        return True

    def generate_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        diffs = [1 + 3*i for i in range(k)]
        for i in range(n):
            for d in diffs:
                j = (i + d) % n
                if i != j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    adj, n = generate_andrasfai(9)
    if not is_valid(adj, n):
        return {'n': 26, 'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]}
    
    best_adj = adj[:]
    best_n = n

    # Greedy Vertex Addition
    # Try to add vertices 27, 28, ... up to 40
    for new_n in range(n + 1, 41):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try to add vertex new_n-1
        # It must connect to an independent set of existing vertices
        # To be efficient, we can try random independent sets or specific ones
        # But checking all independent sets is expensive.
        # Instead, we can try to add the vertex with a random subset of neighbors
        # that forms an independent set.
        
        found = False
        # Limit attempts per vertex to avoid timeout
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and not found:
            attempts += 1
            # Randomly select a subset of existing vertices to be neighbors
            # The neighbors must be independent.
            # Let's pick a random independent set from the current graph.
            # Greedy independent set:
            rem = (1 << n) - 1
            ind_set = []
            while rem:
                # Pick random vertex from rem
                # To make it fast, pick first one? No, random is better for diversity.
                # But we need to ensure the set is independent.
                # Let's just pick a random vertex, add it, remove its neighbors.
                # This gives a maximal independent set, not necessarily all possibilities.
                # For now, let's try this.
                vertices_in_rem = [i for i in range(n) if rem & (1 << i)]
                if not vertices_in_rem: break
                v = random.choice(vertices_in_rem)
                ind_set.append(v)
                rem &= ~(adj[v] | (1 << v))
            
            # Try to add new vertex connected to this independent set
            # But we might want a *smaller* independent set to allow more connections?
            # No, the neighbors must be independent. The size of the neighbor set doesn't matter for triangle-freeness,
            # only that they are mutually non-adjacent.
            # However, a larger neighbor set might make it harder to keep alpha < 10?
            # Actually, connecting to more vertices removes them from independent sets containing the new vertex.
            # So larger neighbor set is better for reducing alpha.
            
            # So, let's use the maximal independent set found.
            new_adj = adj[:] + [0]
            new_n_val = new_n
            for v in ind_set:
                new_adj[v] |= (1 << new_n_val)
                new_adj[new_n_val] |= (1 << v)
            
            # Check validity
            if is_valid(new_adj, new_n_val):
                adj = new_adj
                n = new_n_val
                best_adj = adj[:]
                best_n = n
                found = True
                break
            else:
                # Try a random subset of the independent set?
                # Or just try again with a different random independent set.
                pass

    edges = get_edges_from_adj(best_adj, best_n)
    return {'n': best_n, 'edges': edges}

construct()
