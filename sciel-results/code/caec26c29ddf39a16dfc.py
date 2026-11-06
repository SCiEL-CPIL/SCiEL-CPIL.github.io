import time
import random

def construct():
    start_time = time.time()
    deadline = 145

    def elapsed():
        return time.time() - start_time

    def has_c4(adj_bits, n):
        # adj_bits[i] is an integer bitmask of neighbors of i
        # Check for C4: for each pair (u,v) adjacent, count common neighbors
        # If count >= 2, C4 exists.
        for u in range(n):
            adj_u = adj_bits[u]
            # Iterate over neighbors v of u
            # v must be > u to avoid duplicate checks? No, common neighbors check is symmetric.
            # But we need to check if (u,v) is an edge.
            # Let's just iterate all pairs (u,v) with u < v.
            for v in range(u + 1, n):
                if (adj_u >> v) & 1:
                    # Count common neighbors of u and v
                    common = adj_u & adj_bits[v]
                    # Remove u and v from common (they are not neighbors of themselves)
                    # Actually, adj_u has bit v set, adj_v has bit u set.
                    # common has bit u set if v is neighbor of u? No.
                    # common = N(u) & N(v).
                    # u is in N(v), so bit u is in common.
                    # v is in N(u), so bit v is in common.
                    # We need to exclude u and v.
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if bin(common).count('1') >= 2:
                        return True
        return False

    def min_degree(adj_bits, n):
        return min(bin(adj_bits[i]).count('1') for i in range(n))

    def try_construct(n):
        target = n - 39
        if target < 1:
            target = 1
            
        best_adj = None
        best_md = -1
        
        # Random restarts
        while elapsed() < deadline - 2:
            # Generate random graph
            # Probability p to make it sparse enough to be C4-free likely
            # For n=46, p ~ 0.1-0.2
            p = 0.1 + 0.1 * random.random()
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        
            # Remove C4s
            max_iter = 100
            for _ in range(max_iter):
                if elapsed() > deadline - 2:
                    break
                # Find a C4
                # This is O(n^3) roughly.
                # We can optimize by picking a random edge and checking.
                # Or just scan.
                c4_found = False
                for u in range(n):
                    adj_u = adj_bits[u]
                    for v in range(u + 1, n):
                        if (adj_u >> v) & 1:
                            common = adj_u & adj_bits[v]
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            if bin(common).count('1') >= 2:
                                # Found C4. Remove edge (u,v) or (u,k) or (v,k).
                                # Remove (u,v) is simplest.
                                adj_bits[u] &= ~(1 << v)
                                adj_bits[v] &= ~(1 << u)
                                c4_found = True
                                break
                    if c4_found:
                        break
                if not c4_found:
                    break
            
            # Check if C4 free
            if has_c4(adj_bits, n):
                continue
                
            md = min_degree(adj_bits, n)
            if md > best_md:
                best_md = md
                best_adj = adj_bits[:]
                
            if md >= target:
                break
            
            # Try to add edges to increase min degree
            # Greedy: for each vertex with low degree, try to add edge to non-neighbor
            # that doesn't create C4.
            for _ in range(50):
                if elapsed() > deadline - 2:
                    break
                degs = [bin(adj_bits[i]).count('1') for i in range(n)]
                improved = False
                for v in range(n):
                    if degs[v] < target:
                        # Find non-neighbors
                        non_neighbors = [u for u in range(n) if u != v and not (adj_bits[v] >> u) & 1]
                        random.shuffle(non_neighbors)
                        for u in non_neighbors:
                            # Check if adding (v,u) creates C4
                            # New common neighbors of v and u will be old common + each other?
                            # No, v and u become neighbors.
                            # Common neighbors of v and u in new graph:
                            # Old common neighbors of v and u.
                            # If old common >= 2, adding (v,u) creates C4?
                            # C4: v-a-u-b-v.
                            # Edges v-a, a-u, u-b, b-v.
                            # If a,b are common neighbors of v and u in OLD graph,
                            # then v-a, a-u, u-b, b-v exist.
                            # So if old common >= 2, adding (v,u) creates a C4 (v-u-a-b-v? No, v-a-u-b-v is a cycle of length 4 if v!=a, a!=u, u!=b, b!=v).
                            # Yes.
                            common = adj_bits[v] & adj_bits[u]
                            common &= ~(1 << v)
                            common &= ~(1 << u)
                            if bin(common).count('1') < 2:
                                # Safe to add
                                adj_bits[v] |= (1 << u)
                                adj_bits[u] |= (1 << v)
                                degs[v] += 1
                                degs[u] += 1
                                improved = True
                                break
                if not improved:
                    break
            
            md = min_degree(adj_bits, n)
            if md > best_md:
                best_md = md
                best_adj = adj_bits[:]
                
            if md >= target:
                break

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (best_adj[i] >> j) & 1:
                        edges.append([i, j])
            return n, edges
        return None

    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}
            
    return {'n': 39, 'edges': []}
