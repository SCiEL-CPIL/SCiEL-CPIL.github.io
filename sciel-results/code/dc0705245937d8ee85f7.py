import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # We will try to construct a graph on N vertices.
    # The seed is Paley(37). The known best is Exoo's 42-vertex graph.
    # We start with a 42-vertex graph and try to remove all K5 and I5.
    
    N = 42
    
    # Initialize adjacency matrix using bitmasks for speed.
    # adj[i] is a bitmask of neighbors of i.
    adj = [0] * N
    
    # Helper to check if (u, v) is an edge
    def is_edge(u, v):
        return bool(adj[u] & (1 << v))
    
    # Helper to add/remove edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Helper to count triangles in a subset of vertices given by mask
    # This is used for delta energy calculation.
    def count_triangles_in_mask(mask):
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        count = 0
        n_v = len(verts)
        for i in range(n_v):
            ui = verts[i]
            for j in range(i + 1, n_v):
                uj = verts[j]
                if is_edge(ui, uj):
                    # Common neighbors in mask
                    common = adj[ui] & adj[uj] & mask
                    count += bin(common).count('1')
        return count // 3

    # Helper to count independent triangles in a subset of vertices given by mask
    def count_ind_triangles_in_mask(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        count = 0
        n_v = len(verts)
        for i in range(n_v):
            ui = verts[i]
            for j in range(i + 1, n_v):
                uj = verts[j]
                if not is_edge(ui, uj):
                    # Common non-neighbors in mask
                    # non-adjacent to ui: ~(adj[ui])
                    # non-adjacent to uj: ~(adj[uj])
                    # Intersection: ~(adj[ui] | adj[uj])
                    # Masked: & mask
                    # Exclude ui, uj
                    common_non = ~(adj[ui] | adj[uj]) & mask
                    common_non &= ~(1 << ui)
                    common_non &= ~(1 << uj)
                    count += bin(common_non).count('1')
        return count // 3

    # Delta energy functions
    def delta_k5(u, v, adding):
        if adding:
            # New K5s must contain edge (u,v).
            # Other 3 vertices must be common neighbors of u and v, and form a triangle.
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            return count_triangles_in_mask(common)
        else:
            # Removing edge destroys K5s containing (u,v).
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            return -count_triangles_in_mask(common)

    def delta_i5(u, v, adding):
        # I5 in G is K5 in complement.
        # If adding edge (u,v) to G, we remove it from complement.
        # This destroys I5s containing (u,v) in G? No.
        # I5 in G is a set of 5 vertices with NO edges.
        # If we ADD an edge (u,v) to G, any I5 containing u and v is destroyed.
        # So we subtract the number of I5s containing u and v.
        # An I5 containing u and v requires 3 other vertices that are non-adjacent to u, non-adjacent to v, and mutually non-adjacent.
        # i.e., independent triangles in the common non-neighborhood of u and v.
        
        full_mask = (1 << N) - 1
        # Common non-neighbors of u and v
        # Non-neighbor of u: ~(adj[u])
        # Non-neighbor of v: ~(adj[v])
        # Intersection: ~(adj[u] | adj[v])
        common_non = ~(adj[u] | adj[v]) & full_mask
        common_non &= ~(1 << u)
        common_non &= ~(1 << v)
        
        n_ind_tris = count_ind_triangles_in_mask(common_non)
        
        if adding:
            return -n_ind_tris
        else:
            return n_ind_tris

    # Initialize graph.
    # Strategy: Start with a random graph with density ~0.5, or a known structure.
    # Actually, let's start with the Paley graph on 37 vertices and extend to 42?
    # Or just start random and optimize.
    # Random start is safer to avoid local minima of specific structures.
    
    # Let's try a specific initialization:
    # 1. Paley(37) is (5,5)-free.
    # 2. We have 5 extra vertices.
    # Let's connect the extra vertices to the Paley graph to maintain (5,5)-freeness?
    # This is hard.
    
    # Let's just start with a random graph with p=0.5 and run SA.
    # But SA on 42 vertices might be slow if energy calculation is not fast.
    # Delta energy is O(2^10) roughly? No, count_triangles is O(V^3) in worst case but mask is small.
    # Average degree ~21. Common neighbors ~10. Counting triangles in 10 vertices is fast.
    
    random.seed(42)
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.5:
                add_edge(i, j)
                
    # Calculate initial energy
    # Count all K5 and I5
    # This is expensive. C(42, 5) = 850,668.
    # Checking each is O(1) with bitmasks?
    # Check if 5-set is clique:
    # for u in set: if not (adj[u] & mask_of_others): return False
    # This is fast enough for one pass? 850k * 5 ops = 4M ops. Yes.
    
    def is_clique_5(verts):
        mask = 0
        for v in verts:
            mask |= (1 << v)
        for v in verts:
            if not (adj[v] & (mask ^ (1 << v))):
                return False
        return True
        
    def is_independent_5(verts):
        mask = 0
        for v in verts:
            mask |= (1 << v)
        for v in verts:
            if (adj[v] & (mask ^ (1 << v))):
                return False
        return True
        
    def count_k5():
        c = 0
        for quad in combinations(range(N), 5):
            if is_clique_5(quad):
                c += 1
        return c
        
    def count_i5():
        c = 0
        for quad in combinations(range(N), 5):
            if is_independent_5(quad):
                c += 1
        return c
        
    current_energy = count_k5() + count_i5()
    best_energy = current_energy
    best_adj = adj[:]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 200000 # Adjust based on speed
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_e = is_edge(u, v)
        adding = not is_e
        
        d_k5 = delta_k5(u, v, adding)
        d_i5 = delta_i5(u, v, adding)
        delta = d_k5 + d_i5
        
        new_energy = current_energy + delta
        
        if new_energy < current_energy or random.random() < pow(2.718281828, -delta / T):
            # Accept
            if adding:
                add_edge(u, v)
            else:
                remove_edge(u, v)
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj[:]
                if current_energy == 0:
                    break
        else:
            pass # Reject
            
        T = max(T_min, T * cooling)
        
    # Reconstruct edges from best_adj
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
                
    # Verify validity
    # If best_energy > 0, it's invalid.
    # We must return a valid graph.
    # If we fail to find a 42-vertex valid graph, we should fallback to a smaller one.
    # The seed returns 37.
    # Can we try 41, 40...?
    # Given time, let's just fallback to Paley(37) if 42 fails.
    # Or maybe try to repair? No, too complex.
    
    # Quick check: is best_adj valid?
    # We trust the energy calc if it's 0.
    # But let's double check a few random 5-sets to be safe?
    # No, energy 0 implies valid.
    
    if best_energy > 0:
        # Fallback to Paley(37)
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        n = q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return n, edges
        
    return N, edges

# EVOLVE-BLOCK-END
