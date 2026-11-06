import time
import random

# Triangle-free graphs without independent 10-sets (Ramsey number R(3,10))
# The goal is to find a graph G on n vertices such that:
# 1. G is triangle-free (no K3)
# 2. G has no independent set of size 10 (alpha(G) < 10)
# Maximize n. Known lower bound is 39. Upper bound is 41.

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin for 120s limit

    def time_check():
        return time.time() - start_time

    def has_triangle(adj, n):
        """Check if graph with adjacency bitmasks has a triangle."""
        for u in range(n):
            # Neighbors of u
            neighbors_u = adj[u]
            # For each neighbor v of u
            v_mask = neighbors_u
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                # Check if v has a common neighbor with u
                if adj[v] & neighbors_u:
                    return True
                v_mask ^= v_bit
        return False

    def has_independent_set(adj, n, target=10):
        """
        Check if there is an independent set of size >= target.
        Uses branch and bound on the complement graph (finding a clique of size target).
        Returns True if found, False otherwise.
        """
        # Build complement adjacency
        # comp[u] has bits set for v where u and v are NOT adjacent in G (and u != v)
        comp = [0] * n
        full_mask = (1 << n) - 1
        for u in range(n):
            # ~adj[u] gives non-neighbors, but we need to mask out u itself and bits >= n
            comp[u] = full_mask ^ adj[u] ^ (1 << u)
        
        # Branch and Bound for Max Clique in Complement
        # We just need to find if a clique of size 'target' exists.
        found = [False]
        
        # Heuristic for vertex ordering: sort by degree in complement?
        # Or just use bit ordering. For speed, simple bit ordering is often fine if pruning is good.
        # Let's use a simple recursive search.
        
        def dfs(candidates, size):
            if found[0]:
                return
            if size >= target:
                found[0] = True
                return
            
            if not candidates:
                return
            
            # Upper bound: size + popcount(candidates)
            if size + candidates.bit_count() < target:
                return
            
            # Pick a vertex. 
            # Heuristic: Pick vertex with highest degree in the candidate set to prune?
            # Or just lsb?
            # Let's try lsb for simplicity first, it's fast.
            # If that's too slow, we can switch to degree-based.
            
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Include v
            new_candidates = candidates & comp[v]
            dfs(new_candidates, size + 1)
            
            if found[0]:
                return
            
            # Exclude v
            dfs(candidates ^ v_bit, size)

        dfs(full_mask, 0)
        return found[0]

    def is_valid(adj, n):
        if has_triangle(adj, n):
            return False
        if has_independent_set(adj, n, 10):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def try_circulant(n, start_time):
        """
        Search for a valid circulant graph.
        A circulant graph on n vertices is defined by a set of offsets S.
        u ~ v iff (v - u) % n in S or (u - v) % n in S.
        For symmetry, S = -S mod n.
        """
        # We want to maximize density to reduce alpha, but maintain triangle-free.
        # Triangle-free condition for circulant:
        # If a, b in S, then (b-a) mod n should NOT be in S.
        # Also (a+b) mod n should not be in S? 
        # Triangle i, i+a, i+b. Edges: (i, i+a) -> a in S. (i, i+b) -> b in S. (i+a, i+b) -> b-a in S.
        # So if a,b in S and b-a in S, triangle exists.
        
        max_off = n // 2
        # Try random symmetric sets S
        # Start with high density?
        # Known good structures often have density ~ n/6 or similar.
        
        # Let's try a few specific structured sets first?
        # Or just random search.
        
        # To be efficient, we can pre-filter S for triangle-free condition.
        
        for attempt in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Random S
            S = set()
            # Bias towards including small offsets?
            # Or uniform?
            for off in range(1, max_off + 1):
                if random.random() < 0.25: # ~25% density
                    S.add(off)
                    S.add(n - off)
            
            if not S:
                continue
            
            # Check triangle-free
            valid_tf = True
            S_list = list(S)
            # Optimization: only need to check pairs
            for a in S_list:
                for b in S_list:
                    if (b - a) % n in S:
                        valid_tf = False
                        break
                if not valid_tf:
                    break
            
            if not valid_tf:
                continue
            
            # Build adj
            adj = [0] * n
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj[u] |= (1 << v)
            
            # Check independence
            if not has_independent_set(adj, n, 10):
                return edges_from_adj(adj, n)
        
        return None

    def try_local_search(n, start_time):
        """
        Randomized local search.
        1. Start with random triangle-free graph.
        2. Try to add edges that don't create triangles.
        3. If alpha >= 10, try to swap edges or remove edges to allow new additions?
        """
        # Initialize with a random triangle-free graph
        # Strategy: Start with empty graph, add random edges if triangle-free.
        # This results in a relatively sparse graph.
        # Then try to add more.
        
        adj = [0] * n
        # Random permutation of edges
        edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_list)
        
        # Phase 1: Greedy add
        for u, v in edges_list:
            if adj[u] & (1 << v):
                continue
            if adj[u] & adj[v]:
                continue # Would create triangle
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check if valid
        if not has_independent_set(adj, n, 10):
            return edges_from_adj(adj, n)
        
        # Phase 2: Local Search
        # Try to improve.
        # We want to reduce alpha. Adding edges is the main way.
        # But we are stuck if no safe edges left.
        # We can try removing an edge to open up space for two new edges?
        
        # List of current edges
        current_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    current_edges.add((u, v))
        
        # List of non-edges that are "safe" (don't form triangle)
        # We can't add them yet because we are stuck? No, we added all safe ones in Phase 1.
        # So we must remove edges.
        
        # Heuristic: Remove an edge (u,v) that is part of many "potential" triangles?
        # Or just random.
        
        for step in range(2000):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            if not current_edges:
                break
            
            # Pick a random edge to remove
            u, v = random.choice(list(current_edges))
            
            # Remove it
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edges.discard((u, v))
            
            # Try to add new edges
            # Candidates: edges involving u or v, or others that became safe?
            # Actually, removing (u,v) makes (u, w) and (v, w) potentially safer if w was blocked by u~v?
            # No, triangle is u-v-w. If u-v removed, u-w and v-w don't form triangle with each other unless u~w and v~w and u~v.
            # So removing u-v allows u-w if v~w? No.
            # Triangle u,v,w exists if u~v, v~w, w~u.
            # If we remove u~v, then u~w and v~w are safe from EACH OTHER forming a triangle?
            # They form triangle u-v-w only if u~v is present.
            # So removing u~v allows us to add u~w (if not present) and v~w (if not present) provided no other triangles.
            
            # Let's try to add edges incident to u or v first.
            added_any = False
            for w in range(n):
                if w == u or w == v:
                    continue
                # Try add u-w
                if not (adj[u] & (1 << w)):
                    if not (adj[u] & adj[w]):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.add((min(u,w), max(u,w)))
                        added_any = True
                # Try add v-w
                if not (adj[v] & (1 << w)):
                    if not (adj[v] & adj[w]):
                        adj[v] |= (1 << w)
                        adj[w] |= (1 << v)
                        current_edges.add((min(v,w), max(v,w)))
                        added_any = True
            
            if added_any:
                # Check validity
                if not has_independent_set(adj, n, 10):
                    return edges_from_adj(adj, n)
            else:
                # If we couldn't add anything, revert removal?
                # Or keep it? If we keep it, we have fewer edges, higher alpha likely.
                # Better to revert if no gain.
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        return None

    # Main Logic
    best_n = 0
    best_edges = []
    
    # Try n=40, 39, 38...
    for n in range(40, 25, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # 1. Try Circulant
        edges = try_circulant(n, start_time)
        if edges:
            best_n = n
            best_edges = edges
            break
            
        # 2. Try Local Search (a few times)
        for attempt in range(10):
            if time.time() - start_time > DEADLINE - 5:
                break
            edges = try_local_search(n, start_time)
            if edges:
                best_n = n
                best_edges = edges
                break
                
        if best_n > 0:
            break

    # Fallback: Andrásfai graph And(8) on 25 vertices
    # And(k) has 3k+1 vertices. k=8 -> 25 vertices. Alpha = k+1 = 9.
    # 9 < 10, so it is valid.
    if best_n == 0:
        n = 25
        adj = [0] * n
        # Quadratic residues mod 25
        # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=0, 6^2=11, 7^2=24, 8^2=14, 9^2=6, 10^2=0
        # Non-zero QR: {1, 4, 6, 9, 11, 14, 16, 19, 21, 24}
        # Wait, 19 = 7^2? 49 = 2*25 - 1 = -1. Yes.
        # 21 = 6^2? 36 = 11. No.
        # Let's list properly.
        qrs = set()
        for i in range(1, n):
            qrs.add((i * i) % n)
        qrs.discard(0)
        
        for u in range(n):
            for d in qrs:
                v = (u + d) % n
                if u < v:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Verify
        if is_valid(adj, n):
            best_n = n
            best_edges = edges_from_adj(adj, n)
        else:
            # Should not happen, but fallback to n=1
            best_n = 1
            best_edges = []

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
