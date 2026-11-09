import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Precompute nothing globally, but we will compute path4 masks dynamically
    # as the graph grows.

    def _precompute_path4(v, adj_bits):
        """
        Compute path4_mask[c][i] for subgraph on vertices 0..v-1.
        adj_bits[c][i] is a bitmask of neighbors of i in color c.
        path4_mask[c][i] is a bitmask of vertices d reachable from i 
        via a simple path of length 4 in color c.
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                neighbors_i = adj_bits[c][i]
                if not neighbors_i:
                    continue
                
                mask = 0
                # Path: i -> a -> b -> e -> d
                # Iterate a in neighbors_i
                a = neighbors_i
                while a:
                    # Get lowest set bit
                    lb_a = a & (-a)
                    a ^= lb_a
                    # a is now the vertex index? No, lb_a is the bit.
                    # We need the index.
                    # It's easier to iterate indices if v is small, 
                    # but v can be 17. Bitmask iteration is faster.
                    # Let's stick to index iteration for clarity and correctness first, 
                    # then optimize if needed. v <= 17 is small.
                    pass
                
                # Optimized bitmask version for path length 4
                # We want d such that exists a,b,e:
                # i~a, a~b, b~e, e~d
                # and all distinct.
                
                # Let's use a simpler recursive/iterative expansion with bitmasks
                # Step 1: Neighbors of i (color c)
                # Step 2: Neighbors of those (excluding i)
                # Step 3: Neighbors of those (excluding i, a)
                # Step 4: Neighbors of those (excluding i, a, b)
                
                # This is tricky with bitmasks to ensure "simple path" (distinct vertices).
                # Given v <= 17, we can afford O(v^4) or O(v^5) per vertex if optimized, 
                # but v^4 * 4 colors * v vertices * v steps = v^6 which is 17^6 ~ 24M. 
                # This is too slow if done naively in Python for every backtracking step.
                
                # However, we only recompute this when we add a new vertex v to the main graph,
                # NOT for every backtracking step inside _assign_vertex.
                # In the current design, _assign_vertex(v) calls _precompute_path4(v, ...).
                # This happens n times (17 times).
                # So cost is 17 * (cost of precompute).
                # Cost of precompute for fixed v:
                # For each c (4), for each i (v), find all d.
                # Naive loop: 4 * v * (sum over paths).
                # Max paths of length 4 in K_17: 17*16*15*14*13 ~ 6.5M.
                # 4 * 17 * 6.5M is too high? No, we only count valid simple paths.
                # Actually, the number of simple paths of length 4 in a dense graph is high.
                
                # Let's try a slightly different approach:
                # For each pair (i, d), check if there is a path of length 4.
                # This is equivalent to checking if (A_c^4)[i][d] > 0, 
                # but A^4 counts walks, not simple paths.
                # However, for C6 detection, we need simple paths.
                # A walk of length 4 might revisit vertices.
                # If a walk revisits a vertex, it's not a simple path.
                # But if there is a simple path, it's a subset of walks.
                # If A_c^4[i][d] > 0, does it imply a simple path?
                # Not necessarily. E.g., i-a-i-b-d is a walk but not simple.
                # However, if the graph is dense, it's likely.
                # But we need exactness.
                
                # Let's stick to the explicit path enumeration but optimize with bitmasks.
                # For fixed i, c:
                # 1. Get neighbors N1 = adj_bits[c][i]
                # 2. For each a in N1:
                #    Get N2 = adj_bits[c][a] & ~bit(i)
                # 3. For each b in N2:
                #    Get N3 = adj_bits[c][b] & ~(bit(i) | bit(a))
                # 4. For each e in N3:
                #    Get N4 = adj_bits[c][e] & ~(bit(i) | bit(a) | bit(b))
                #    mask |= N4
                
                # This is O(deg(i) * deg(a) * deg(b) * deg(e)).
                # In K_17, deg is 16. 16^4 = 65536 operations per (i, c).
                # 17 * 4 * 65536 ~ 4.5M operations.
                # Python can do ~10M-50M simple ops/sec.
                # So one precompute takes ~0.1s - 0.5s.
                # 17 precomputes -> ~2-8s.
                # This is acceptable.
                
                # Let's implement this bitmask loop.
                mask = 0
                # Iterate a
                a_mask = adj_bits[c][i]
                while a_mask:
                    lb_a = a_mask & (-a_mask)
                    a_mask ^= lb_a
                    a_idx = lb_a.bit_length() - 1
                    
                    # Iterate b
                    b_mask = adj_bits[c][a_idx] & ~lb_a & ~(1 << i)
                    while b_mask:
                        lb_b = b_mask & (-b_mask)
                        b_mask ^= lb_b
                        b_idx = lb_b.bit_length() - 1
                        
                        # Iterate e
                        e_mask = adj_bits[c][b_idx] & ~lb_a & ~lb_b & ~(1 << i)
                        while e_mask:
                            lb_e = e_mask & (-e_mask)
                            e_mask ^= lb_e
                            e_idx = lb_e.bit_length() - 1
                            
                            # d is in adj_bits[c][e_idx] excluding i, a, b
                            d_mask = adj_bits[c][e_idx] & ~lb_a & ~lb_b & ~lb_e & ~(1 << i)
                            mask |= d_mask
                            
                path4_mask[c][i] = mask

        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        
        # adj_bits[c][i] is a bitmask of neighbors of i in color c
        adj_bits = [[0] * n_target for _ in range(4)]
        
        # We will store the final colors in a list of lists or a flat list
        # For now, we just need to know if a solution exists.
        # We'll reconstruct colors at the end if needed, but the task asks for colors.
        # We'll store colors in a 2D array or flat list.
        colors = [[-1] * n_target for _ in range(n_target)]

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            # This uses adj_bits which is up to date for vertices 0..v-1
            path4_mask = _precompute_path4(v, adj_bits)

            # Backtrack over edges (v, 0), (v, 1), ..., (v, v-1)
            # masks[c] holds bits for vertices i < current_idx assigned color c to (v, i)
            masks = [0, 0, 0, 0]
            
            # Heuristic: Order of vertices 0..v-1 to assign?
            # Fixed order 0, 1, ..., v-1 is standard.
            # Randomizing the order of edges might help break symmetry.
            # Let's try fixed order first for simplicity and determinism within a seed.
            
            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                # Try each color for edge (v, idx)
                # Shuffle colors? Or fixed?
                # Fixed 0,1,2,3 is fine.
                for c in range(4):
                    # Check constraint: 
                    # If we assign color c to (v, idx), then idx must not be 
                    # connected via Path4 to any vertex j already assigned color c.
                    # path4_mask[c][idx] has bits for all j such that path4 exists.
                    # masks[c] has bits for j already assigned color c.
                    # Intersection must be empty.
                    if path4_mask[c][idx] & masks[c]:
                        continue
                    
                    # Assign
                    masks[c] |= (1 << idx)
                    colors[idx][v] = c
                    colors[v][idx] = c
                    adj_bits[c][idx] |= (1 << v)
                    adj_bits[c][v] |= (1 << idx)

                    if _backtrack(idx + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << idx)
                    colors[idx][v] = -1
                    colors[v][idx] = -1
                    adj_bits[c][idx] &= ~(1 << v)
                    adj_bits[c][v] &= ~(1 << idx)

                return False

            return _backtrack(0)

        # Try to build the coloring vertex by vertex
        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            
            if not _assign_vertex(v):
                return None

        # Extract colors in lexicographic order
        res_colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                res_colors.append(colors[i][j])
        return res_colors

    def _verify(colors, n):
        """Verify no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1

        # Check for C6 in each color
        # For each color c, check all subsets of 6 vertices
        # This is O(C(n,6) * 60). C(17,6) = 12376. 12376 * 60 * 4 ~ 3M. Fast.
        from itertools import combinations, permutations
        
        # Precompute cycles
        cycles = []
        seen = set()
        for perm in permutations(range(1, 6)):
            cycle = (0,) + perm
            edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
            if edges not in seen:
                seen.add(edges)
                cycles.append(cycle)

        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    # 17 is the target.
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break

        found = False
        # Try multiple seeds
        # Increase seeds if time permits
        max_seeds = 50 if n == 17 else 20
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break

            colors = _search(n, seed, deadline - 10)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    # Should not happen if logic is correct
                    continue

        if found:
            break

    if best_n == 0:
        # Fallback to a known small valid coloring
        # n=9 pattern: (j-i) % 4
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        
        # Verify fallback
        if _verify(colors, n):
            best_n = n
            best_colors = colors
        else:
            # Extremely unlikely
            n = 5
            best_n = n
            best_colors = [0] * (n*(n-1)//2)

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Found n={res['n']}")
