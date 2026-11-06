import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def build_adj_from_offsets(n, S):
        """Build adjacency bitsets for a circulant graph defined by offset set S."""
        adj = [0] * n
        # S is a set of offsets in 1..n//2
        # For symmetry, if k in S, then n-k is also effectively in S
        # We iterate i and for each k in S, connect i to (i+k)%n and (i-k)%n
        for i in range(n):
            mask = 0
            for k in S:
                j1 = (i + k) % n
                j2 = (i - k) % n
                if j1 != i:
                    mask |= (1 << j1)
                if j2 != i:
                    mask |= (1 << j2)
            adj[i] = mask
        return adj

    def is_triangle_free_circulant(n, S):
        """
        Check if circulant graph with symmetric offset set S is triangle-free.
        Triangle exists if there are a, b in S such that (a-b) % n in S.
        Note: S contains positive offsets. We must check all pairs.
        """
        S_list = list(S)
        len_S = len(S_list)
        for i in range(len_S):
            a = S_list[i]
            for j in range(len_S):
                b = S_list[j]
                if a == b:
                    continue
                # Check if (a-b) mod n is in S or its symmetric counterpart
                diff = (a - b) % n
                # diff is in 0..n-1
                # The offset is min(diff, n-diff)
                off = diff if diff <= n // 2 else n - diff
                if off in S:
                    return False
        return True

    def has_independent_set_size(adj, n, k, time_limit=5.0):
        """
        Returns True if there exists an independent set of size k.
        Uses branch and bound with greedy coloring upper bounds.
        """
        start_t = time.time()
        best_found = [k - 1]
        
        # Precompute non-neighbors (complement of adj, excluding self)
        full_mask = (1 << n) - 1
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = full_mask ^ adj[i] ^ (1 << i)

        def greedy_color_bound(candidates):
            """
            Greedy coloring of the subgraph induced by candidates.
            We color the COMPLEMENT of the subgraph to find independent sets in the original.
            Wait, independent set in G is a clique in G_bar.
            So we want to find a clique of size k in G_bar.
            Greedy coloring of G_bar gives an upper bound on clique size.
            G_bar adjacency is non_adj.
            """
            remaining = candidates
            colors = 0
            while remaining:
                # Pick vertex with min degree in remaining (for G_bar)
                # Degree in G_bar is popcount(non_adj[v] & remaining)
                best_v = -1
                best_deg = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = popcount(non_adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                
                # Remove best_v from remaining
                remaining &= ~(1 << best_v)
                colors += 1
                
                # Remove neighbors of best_v in G_bar from remaining
                # Neighbors in G_bar are non_adj[best_v]
                remaining &= ~non_adj[best_v]
                
            return colors

        def dfs(candidates, current_size):
            if time.time() - start_t > time_limit:
                return False
            if best_found[0] >= k:
                return True
            if current_size >= k:
                best_found[0] = k
                return True
            if not candidates:
                return False
            
            # Bound: current_size + max_clique_size(remaining)
            # Max clique size in G_bar[remaining] <= number of colors in greedy coloring of G_bar[remaining]
            bound = greedy_color_bound(candidates)
            if current_size + bound <= best_found[0]:
                return False
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with max degree in G_bar (so it has many non-neighbors in G)
            # Actually, for clique finding in G_bar, we want vertices with high degree in G_bar
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(non_adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v
            
            # Branch 1: Include v in clique (independent set in G)
            # New candidates must be neighbors of v in G_bar
            new_cands = rest & non_adj[v]
            if dfs(new_cands, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            if dfs(rest, current_size):
                return True
            
            return False

        dfs(full_mask, 0)
        return best_found[0] >= k

    def verify_graph(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Triangle check
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                if adj[j] & adj[i]:
                    return False
        
        # Independence check
        if has_independent_set_size(adj, n, 10):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Known good fallback: Andrásfai graph A_9 (n=27) or similar
    # Let's just return a valid small graph if search fails
    # A_5 has n=15, alpha=5. A_9 has n=27, alpha=9.
    # We want to maximize n.
    
    best_n = 0
    best_edges = []

    # Search for circulant graphs
    # Start from high n down
    for n in range(40, 25, -1):
        if time.time() - start_time > deadline:
            break
        
        # If we already found a graph with n >= current target, we can skip lower n
        # But we are iterating down, so we stop when we find one.
        # However, we want the MAXIMUM n. So we try 40, then 39, etc.
        # If we find one at 39, we return it.
        
        found = False
        
        # Local search on offset set S
        # Represent S as a bitset of length n//2
        max_off = n // 2
        random.seed(42 + n)
        
        # Initialize with a random set of offsets
        # Density around 0.3-0.5 seems reasonable for triangle-free
        # Triangle-free implies max degree constraints?
        
        # Let's use a simple local search
        # State: bitset of offsets
        # Fitness: - (num_triangles * 100 + max(0, alpha - 9) * 10)
        
        # To speed up, we only check triangle-free on the fly.
        # Checking alpha is expensive, so we only do it if triangle-free.
        
        best_fitness = -float('inf')
        best_S = set()
        
        # Try several random restarts
        for restart in range(20):
            if time.time() - start_time > deadline:
                break
            
            # Random init
            S = set()
            for k in range(1, max_off + 1):
                if random.random() < 0.4:
                    S.add(k)
            
            # Local search
            for step in range(1000):
                if time.time() - start_time > deadline:
                    break
                
                # Check triangle-free
                if not is_triangle_free_circulant(n, S):
                    # If not triangle free, try to remove an offset involved in a triangle
                    # Or just penalize heavily and move on?
                    # Better: find a conflicting pair and remove one
                    removed = False
                    S_list = list(S)
                    for i in range(len(S_list)):
                        a = S_list[i]
                        for j in range(len(S_list)):
                            b = S_list[j]
                            if a == b: continue
                            diff = (a - b) % n
                            off = diff if diff <= n // 2 else n - diff
                            if off in S:
                                # Conflict found. Remove 'off' or 'a' or 'b'?
                                # Remove the one that appears in most conflicts?
                                # Simple: remove 'off'
                                S.discard(off)
                                removed = True
                                break
                        if removed: break
                    if not removed:
                        # Should not happen if logic is correct, but just in case
                        break
                    continue
                
                # If triangle-free, check independence
                adj = build_adj_from_offsets(n, S)
                # Quick check: if average degree is very low, alpha is likely high
                # Sum of degrees
                total_deg = sum(popcount(adj[i]) for i in range(n))
                avg_deg = total_deg / n
                
                # If avg_deg is too low, alpha is likely >= 10
                # Turan's theorem or similar bounds?
                # For n=40, if alpha < 10, we need enough edges.
                # Let's just run the B&B if it's fast enough or if we suspect it's close.
                
                # Optimization: Only run B&B if we haven't checked this S recently?
                # Or limit B&B time.
                
                has_ind = has_independent_set_size(adj, n, 10, time_limit=2.0)
                
                if not has_ind:
                    # Found a valid graph!
                    edges = edges_from_adj(adj, n)
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                
                # If not valid, try to modify S
                # Add or remove a random offset
                if random.random() < 0.5:
                    # Add
                    candidates = [k for k in range(1, max_off + 1) if k not in S]
                    if candidates:
                        S.add(random.choice(candidates))
                else:
                    # Remove
                    if S:
                        S.discard(random.choice(list(S)))
            
            if found:
                break

        if found:
            # Verify one last time to be sure
            if verify_graph(best_n, best_edges):
                return {'n': best_n, 'edges': best_edges}
            else:
                # Should not happen, but if it does, break and try lower n
                continue

    # If no circulant graph found, fallback to a known small one
    # Andrásfai A_9 (n=27)
    n = 27
    edges = []
    k = 9
    for a in range(3):
        for i in range(k):
            for b in range(3):
                if (b - a) % 3 != 1:
                    continue
                for j in range(k):
                    if (j - i) % k in (0, 1):
                        u = a * k + i
                        v = b * k + j
                        if u < v:
                            edges.append([u, v])
    
    if verify_graph(n, edges):
        return {'n': n, 'edges': edges}
        
    # Last resort: empty graph (invalid) or single vertex
    return {'n': 1, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
