import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False

    def find_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        d_bit = d_candidates & (-d_candidates)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    def max_triangle_free_size(adj, time_limit=3.0):
        """
        Find the size of the largest triangle-free induced subgraph.
        Uses exact branch and bound with bitmasks.
        """
        best = [0]
        t0 = time.time()
        timeout = [False]

        # Precompute degrees for ordering
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Order vertices: High degree first (more constraints earlier)
        order = sorted(range(n), key=lambda v: -deg[v])

        # Map original vertex to index in order
        # We iterate through 'order' indices, but need to access adj[orig_v]
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pruning: even if we add all remaining, can we beat best?
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: Include v
            if not timeout[0]:
                # Check if adding v creates a triangle
                # Triangle (v, u, w) exists if u, w in current_mask, u~v, w~v, u~w.
                # Let N_v = neighbors of v.
                # Let C = current_mask & N_v.
                # If the induced subgraph on C has any edge (u, w), then (v, u, w) is a triangle.
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                valid = True
                if C:
                    # Check if there is any edge within C
                    # Iterate vertices in C
                    tmp_c = C
                    while tmp_c:
                        bit = tmp_c & (-tmp_c)
                        u = bit.bit_length() - 1
                        tmp_c ^= bit
                        # Check if u has any neighbor in C
                        # adj[u] & C checks neighbors of u.
                        # Since C only contains neighbors of v, and u is in C,
                        # if adj[u] & C is non-zero, there is a neighbor w in C.
                        # (u, w) is an edge. (v, u) and (v, w) are edges by definition of C.
                        # So (v, u, w) is a triangle.
                        if adj[u] & C:
                            valid = False
                            break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    def construct_paley_17():
        """
        Construct Paley(17) graph on vertices 0..16.
        Vertex 17 is isolated.
        """
        adj = [0] * n
        # Quadratic residues modulo 17:
        # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj

    def make_k4_free(adj):
        """
        Remove edges to make graph K4-free.
        Returns True if successful.
        """
        for _ in range(1000):
            k4 = find_k4(adj)
            if k4 is None:
                return True
            a, b, c, d = k4
            # Remove a random edge from the K4
            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            x, y = pair
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        return not has_k4(adj)

    # 1. Initialize with Paley(17) + isolated vertex
    adj = construct_paley_17()
    
    # Verify it's K4-free (it should be, Paley(17) is K4-free)
    if has_k4(adj):
        make_k4_free(adj)
    
    current_score = max_triangle_free_size(adj, time_limit=3.0)
    
    # 2. Local Search: Add edges to reduce triangle-free size
    # We want to minimize the size of the largest triangle-free induced subgraph.
    # Adding edges generally increases triangle density, which can reduce the size of triangle-free sets.
    # However, we must maintain K4-free property.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Generate list of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u, v) creates a K4
            # A K4 is created if there exists a pair (w, x) such that
            # w~u, w~v, x~u, x~v, and w~x.
            # i.e., (w, x) is an edge in the common neighborhood of u and v.
            common = adj[u] & adj[v]
            # Exclude u and v from common just in case (though they aren't neighbors of themselves)
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            creates_k4 = False
            # Check if subgraph induced by common has any edge
            tmp_c = common
            while tmp_c:
                bit = tmp_c & (-tmp_c)
                w = bit.bit_length() - 1
                tmp_c ^= bit
                if adj[w] & common:
                    creates_k4 = True
                    break
            
            if not creates_k4:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_score = max_triangle_free_size(adj, time_limit=2.0)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Restart loop with new non-edges
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # If no improvement in this pass, we could try removing edges or swapping, 
        # but adding edges is the primary driver for reducing triangle-free sets in this context.
        # If we get stuck, we might try a random restart or different initialization, 
        # but Paley(17) is a strong baseline.
        
        # Optional: Try edge swaps if stuck? 
        # For now, simple addition is the main strategy.
        
        # To avoid infinite loops if no improvement is possible but `improved` is False,
        # the while loop terminates.
        
        # If we want to be more aggressive, we could also try removing edges that don't help,
        # but that usually increases the triangle-free set size (or keeps it same), so it's counterproductive 
        # unless it allows adding other edges later.
        
        # Let's add a simple random restart mechanism if we get stuck for a long time?
        # No, keep it simple. Paley(17) is known to be good.
        
        # If we haven't improved, we might try a few random edge additions even if they don't immediately improve,
        # hoping to find a better configuration later? This is risky for time.
        # Let's stick to greedy improvement.
        
        # One more thing: The isolated vertex 17. 
        # In Paley(17)+isolated, the isolated vertex is always in any triangle-free set (since it has no edges).
        # So the max triangle-free size is 1 + max_triangle_free_size(Paley(17)).
        # Can we do better by connecting 17?
        # If we connect 17 to some vertices, we might break large triangle-free sets in Paley(17) 
        # if 17 is part of a triangle? No, 17 has degree 0 initially.
        # If we add edges from 17, we create triangles involving 17.
        # This might force the removal of vertices from the triangle-free set.
        # Let's see if the local search handles this.
        # The non_edges list includes (17, v) for all v.
        # So the local search will try to add edges from 17.
        # If adding (17, v) reduces the max triangle-free size, it will be accepted.
        
        # Let's run for a bit longer to allow multiple passes.
        # The `improved` flag breaks the inner loop, then the outer loop continues.
        # This allows sequential improvements.

    return edges_from_adj(adj)

# Dummy main to ensure no syntax errors if run directly, though construct() is the entry point
if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
