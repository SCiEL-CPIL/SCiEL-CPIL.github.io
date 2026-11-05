import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # Safety margin under 120s kill
    rng = random.Random(12345)

    def has_k4(n, A):
        """Check if graph has any K4. Used for initial validation."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def find_k4(n, A):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def find_ind_k(n, A, k):
        """Find an independent set of size k using Bron-Kerbosch on complement.
        Returns the set as a tuple of k vertices, or None if max independent set < k."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            nonlocal found
            if found[0] is not None:
                return
            if len(R) == k:
                found[0] = tuple(R)
                return
            if not P:
                return
            # Bound: if we can't reach k, stop
            if len(R) + bin(P).count("1") < k:
                return
            
            # Pivot selection
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                # Select u in P with max degree in P to minimize candidates
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def max_independent_set_size(n, A):
        """Exact max independent set size via Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count("1") <= best[0]:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return best[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if max_independent_set_size(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Start with Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    n0 = q
    A = [0] * n0
    for i in range(n0):
        for j in range(n0):
            if i != j and (j - i) % q in squares:
                A[i] |= (1 << j)
    assert valid(n0, A), "Paley(17) should be valid"
    best_n = n0
    best_A = list(A)

    def check_add_vertex(cur_n, cur_A, new_idx, neighbors_mask):
        """
        Check if adding vertex new_idx with neighbors defined by neighbors_mask
        maintains K4-free and alpha < 6.
        Returns True if valid.
        """
        # 1. Check K4 constraint:
        # A new K4 must include the new vertex.
        # So N(new) must be triangle-free in G.
        # If there is a triangle (u, v, w) in N(new), then {new, u, v, w} is a K4.
        
        # Iterate over pairs in neighbors to find a triangle
        # For efficiency, we can precompute triangles or just check on the fly.
        # Since n is small (<40), on-the-fly is fine.
        
        # Get list of neighbors
        neigh_list = []
        m = neighbors_mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            neigh_list.append(v)
            m ^= low
            
        # Check for triangles in neigh_list
        # For each pair (u, v) in neigh_list, check if there is w in neigh_list 
        # such that u-v, v-w, w-u are edges in G.
        L = len(neigh_list)
        for i in range(L):
            u = neigh_list[i]
            for j in range(i + 1, L):
                v = neigh_list[j]
                if not ((cur_A[v] >> u) & 1):
                    continue
                # Check if there is a w in neigh_list[j+1:] connected to both u and v
                # Actually, we just need to check if common neighbors of u and v 
                # intersect with neigh_list.
                common = cur_A[u] & cur_A[v] & neighbors_mask
                if common:
                    return False # Triangle found in N(new), creates K4

        # 2. Check Independent Set constraint:
        # A new independent set of size 6 must include the new vertex.
        # So N_bar(new) must contain an independent set of size 5 in G.
        # N_bar(new) = All vertices \ {new} \ N(new)
        
        non_neigh_mask = ((1 << cur_n) - 1) & ~neighbors_mask
        
        # We need to check if G[non_neigh] has an independent set of size 5.
        # This is equivalent to checking if the complement of G[non_neigh] has a clique of size 5.
        # Or, using Bron-Kerbosch on the complement of G[non_neigh] to find max independent set size.
        
        # Extract subgraph induced by non_neigh_mask
        # To use our existing BK function, we need to map these vertices to 0..k-1
        # or modify BK to work on a subset mask.
        
        # Let's implement a quick BK check for alpha >= 5 on the subset.
        # If alpha(G[non_neigh]) >= 5, then adding new vertex creates an independent set of size 6.
        
        # Optimization: If the number of non-neighbors is < 5, it's impossible.
        if bin(non_neigh_mask).count("1") < 5:
            return True # No independent set of size 5 possible in non-neighbors
        
        # Run a specific check for independent set of size 5 in the subgraph defined by non_neigh_mask
        # We can reuse the logic of find_ind_k but restricted to the mask.
        
        # Construct subgraph adjacency for the subset
        sub_verts = []
        m = non_neigh_mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            sub_verts.append(v)
            m ^= low
            
        k_sub = len(sub_verts)
        if k_sub < 5:
            return True
            
        # Map original indices to 0..k_sub-1
        idx_map = {v: i for i, v in enumerate(sub_verts)}
        
        # Build adjacency list for subgraph
        sub_A = [0] * k_sub
        for i in range(k_sub):
            u = sub_verts[i]
            for j in range(i + 1, k_sub):
                v = sub_verts[j]
                if (cur_A[v] >> u) & 1:
                    sub_A[i] |= (1 << j)
                    sub_A[j] |= (1 << i)
                    
        # Check if max_independent_set_size(k_sub, sub_A) >= 5
        # To save time, we can just check if there IS an independent set of size 5.
        if find_ind_k(k_sub, sub_A, 5) is not None:
            return False
        return True

    def try_grow_smart(cur_n, cur_A, deadline):
        """
        Try to grow the graph by 1 vertex using a smart neighborhood selection.
        """
        new_n = cur_n + 1
        if new_n > 40:
            return None
        
        new_idx = cur_n
        
        # Strategy:
        # We want a neighborhood N(new) that is:
        # 1. Triangle-free in G (to avoid K4)
        # 2. Such that G[V \ N(new)] has alpha < 5 (to avoid Ind6)
        
        # Heuristic:
        # Start with a random set of neighbors, then refine.
        # Or, start with empty set and add vertices that don't create triangles,
        # while ensuring we don't leave too many non-neighbors that form an Ind5.
        
        # Let's try a randomized local search on the neighborhood mask.
        
        # Initial guess: connect to ~50% of vertices
        # But we must avoid triangles.
        
        # A better approach:
        # 1. Identify all "safe" vertices to connect to.
        # 2. We can use a greedy algorithm to build N(new).
        
        # Let's try multiple random restarts of a greedy construction.
        
        best_mask = None
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            # Shuffle vertices
            verts = list(range(cur_n))
            rng.shuffle(verts)
            
            current_mask = 0
            # We will try to add vertices to current_mask one by one.
            # When adding vertex v to current_mask:
            # - v must not form a triangle with any pair in current_mask.
            # - This check is: (cur_A[v] & current_mask) must not contain an edge.
            
            # Also, we want to balance the size of N(new) and N_bar(new).
            
            for v in verts:
                # Check if adding v to current_mask creates a triangle in N(new)
                # Triangle in N(new) means there exist u, w in current_mask such that u-v, v-w, w-u are edges.
                # This is equivalent to: (cur_A[v] & current_mask) contains an edge in G.
                
                neighbors_of_v_in_current = cur_A[v] & current_mask
                # Check if neighbors_of_v_in_current has any edge
                has_edge = False
                m = neighbors_of_v_in_current
                while m:
                    low = m & -m
                    u = low.bit_length() - 1
                    m ^= low
                    # Check if u has any neighbor in (neighbors_of_v_in_current without u)
                    if cur_A[u] & (neighbors_of_v_in_current & ~(1 << u)):
                        has_edge = True
                        break
                if has_edge:
                    continue # Adding v would create a triangle in N(new)
                
                # Tentatively add v
                new_mask = current_mask | (1 << v)
                
                # Check if this leads to a valid graph.
                # We need G[V \ new_mask] to have alpha < 5.
                # This check is expensive, so we might do it less frequently or with a bound.
                # For now, let's just build a large triangle-free set and check at the end.
                # But we need to ensure we don't leave too many non-neighbors.
                
                # Heuristic: If we have added enough neighbors, check validity.
                # Target size for N(new) is around cur_n / 2.
                if bin(current_mask).count("1") > cur_n // 2:
                    if check_add_vertex(cur_n, cur_A, new_idx, new_mask):
                        return new_n, new_mask
                    # If invalid, maybe we added too many neighbors? Or not enough?
                    # If alpha(G[V\N]) >= 5, it means N_bar is too "independent".
                    # Adding more neighbors to N(new) reduces N_bar, which might help.
                    # So we should continue adding neighbors.
                
                current_mask = new_mask
            
            # After processing all vertices, check the final mask
            if current_mask and check_add_vertex(cur_n, cur_A, new_idx, current_mask):
                return new_n, current_mask

        return None

    def local_repair(cur_n, cur_A, rng, deadline):
        """Targeted repair: fix specific K4s or independent sets."""
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            k4 = find_k4(cur_n, cur_A)
            if k4:
                a, b, c, d = k4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                cur_A[u] &= ~(1 << v)
                cur_A[v] &= ~(1 << u)
                continue
            
            ind6 = find_ind_k(cur_n, cur_A, 6)
            if ind6:
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                cur_A[u] |= (1 << v)
                cur_A[v] |= (1 << u)
                continue
            
            break
        
        return cur_A

    def random_flips(cur_n, cur_A, rng, deadline, n_flips=5):
        """Apply some random edge flips."""
        for _ in range(n_flips):
            if time.time() > deadline - 5:
                break
            i = rng.randrange(cur_n)
            j = rng.randrange(cur_n)
            if i == j:
                continue
            if (cur_A[j] >> i) & 1:
                cur_A[i] &= ~(1 << j)
                cur_A[j] &= ~(1 << i)
            else:
                cur_A[i] |= (1 << j)
                cur_A[j] |= (1 << i)
        return cur_A

    # Main search loop
    while time.time() < deadline:
        cur_n = best_n
        cur_A = list(best_A)
        
        # Try to grow
        if cur_n < 40:
            result = try_grow_smart(cur_n, cur_A, deadline)
            if result:
                new_n, new_mask = result
                # Update adjacency list
                new_A = cur_A + [0]
                for i in range(cur_n):
                    if (new_mask >> i) & 1:
                        new_A[i] |= (1 << new_n - 1)
                        new_A[new_n - 1] |= (1 << i)
                
                # Final verification
                if valid(new_n, new_A):
                    best_n = new_n
                    best_A = new_A
                    continue
        
        # If we can't grow, try to perturb and repair to find a different local optimum
        if time.time() < deadline - 20:
            perturbed = random_flips(cur_n, cur_A, rng, deadline, n_flips=rng.randint(1, 5))
            repaired = local_repair(cur_n, perturbed, rng, deadline)
            if not has_k4(cur_n, repaired) and max_independent_set_size(cur_n, repaired) < 6:
                # Try to grow from this new state
                result = try_grow_smart(cur_n, repaired, deadline)
                if result:
                    new_n, new_mask = result
                    new_A = repaired + [0]
                    for i in range(cur_n):
                        if (new_mask >> i) & 1:
                            new_A[i] |= (1 << new_n - 1)
                            new_A[new_n - 1] |= (1 << i)
                    if valid(new_n, new_A):
                        best_n = new_n
                        best_A = new_A

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
# EVOLVE-BLOCK-END
