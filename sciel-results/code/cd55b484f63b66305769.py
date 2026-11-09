import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Precompute all 60 distinct Hamiltonian cycles on 6 labeled vertices
    # for final verification
    _cycles_cache = None

    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            # Generate unique cycles up to rotation and reflection
            # Fix vertex 0 at start. Permute 1..5.
            # 5! = 120 permutations. / 2 for reflection = 60 cycles.
            for perm in __import__('itertools').permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache

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

        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)

        cycles = _get_cycles()

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

    def _precompute_path4_bitmask(v, adj_bitmasks):
        """
        Compute path4_mask[c][i] for subgraph on vertices 0..v-1 using bitmasks.
        adj_bitmasks[c][i] is a bitmask of neighbors of i in color c.
        path4_mask[c][i] is a bitmask where bit d is set if there is a 
        simple path of length 4 from i to d in color c graph.
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                mask_i = adj_bitmasks[c][i]
                if mask_i == 0:
                    continue
                
                mask = 0
                # Path: i -> a -> b -> e -> d
                # For each neighbor a of i:
                a_mask = mask_i
                while a_mask:
                    # Get lowest set bit
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    # b must be neighbor of a, not i
                    b_mask = adj_bitmasks[c][a] & ~(1 << i)
                    if b_mask == 0:
                        continue
                    
                    # For each neighbor b of a (excluding i):
                    b_mask_iter = b_mask
                    while b_mask_iter:
                        b_bit = b_mask_iter & (-b_mask_iter)
                        b = b_bit.bit_length() - 1
                        b_mask_iter ^= b_bit
                        
                        # e must be neighbor of b, not i, a
                        e_mask = adj_bitmasks[c][b] & ~(1 << i) & ~(1 << a)
                        if e_mask == 0:
                            continue
                        
                        # For each neighbor e of b (excluding i, a):
                        e_mask_iter = e_mask
                        while e_mask_iter:
                            e_bit = e_mask_iter & (-e_mask_iter)
                            e = e_bit.bit_length() - 1
                            e_mask_iter ^= e_bit
                            
                            # d must be neighbor of e, not i, a, b
                            d_mask = adj_bitmasks[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                            mask |= d_mask
                
                path4_mask[c][i] = mask
        
        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        adj_bitmasks = [[0] * n_target for _ in range(4)]

        def _update_bitmasks(i, j, c, add):
            """Update adj_bitmasks when adding/removing edge (i,j) with color c."""
            if add:
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
            else:
                adj_bitmasks[c][i] &= ~(1 << j)
                adj_bitmasks[c][j] &= ~(1 << i)

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            path4_mask = _precompute_path4_bitmask(v, adj_bitmasks)

            # Backtrack over edges (v, 0), (v, 1), ..., (v, v-1)
            masks = [0, 0, 0, 0]
            
            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                # Try each color for edge (v, idx)
                for c in range(4):
                    # Check constraint: 
                    if path4_mask[c][idx] & masks[c]:
                        continue
                    
                    # Assign
                    masks[c] |= (1 << idx)
                    adj[idx][v] = c
                    adj[v][idx] = c
                    adj_list[c][idx].add(v)
                    adj_list[c][v].add(idx)
                    _update_bitmasks(idx, v, c, True)

                    if _backtrack(idx + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << idx)
                    adj[idx][v] = -1
                    adj[v][idx] = -1
                    adj_list[c][idx].remove(v)
                    adj_list[c][v].remove(idx)
                    _update_bitmasks(idx, v, c, False)

                return False

            return _backtrack(0)

        # Try to build the coloring vertex by vertex
        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            
            if not _assign_vertex(v):
                return None

        # Extract colors in lexicographic order
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return colors

    def _local_search(colors, n, max_iters, deadline):
        """
        Local search polish: try to reduce C6 count by flipping edge colors.
        Returns improved colors or None if no improvement found.
        """
        if time.time() > deadline:
            return None
        
        # Build adjacency from colors
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        def _count_c6s():
            """Count total C6s across all colors."""
            total = 0
            cycles = _get_cycles()
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
                            total += 1
            return total
        
        current_c6s = _count_c6s()
        if current_c6s == 0:
            return colors
        
        best_colors = colors[:]
        best_c6s = current_c6s
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            e = rng.randint(0, n*(n-1)//2 - 1)
            # Find (i,j) for index e
            # e = i*(2n-1-i)//2 + (j-i-1) for i<j
            # Simpler: iterate
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    idx2 = 0
                    for ii in range(i):
                        for jj in range(ii+1, n):
                            idx2 += 1
                    for jj in range(i+1, j):
                        idx2 += 1
                    if idx2 == e:
                        old_c = adj[i][j]
                        new_c = (old_c + 1) % 4
                        # Flip
                        adj[i][j] = new_c
                        adj[j][i] = new_c
                        adj_list[old_c][i].remove(j)
                        adj_list[old_c][j].remove(i)
                        adj_list[new_c][i].add(j)
                        adj_list[new_c][j].add(i)
                        
                        new_c6s = _count_c6s()
                        if new_c6s < best_c6s:
                            best_c6s = new_c6s
                            best_colors = []
                            for ii in range(n):
                                for jj in range(ii+1, n):
                                    best_colors.append(adj[ii][jj])
                        else:
                            # Revert
                            adj[i][j] = old_c
                            adj[j][i] = old_c
                            adj_list[new_c][i].remove(j)
                            adj_list[new_c][j].remove(i)
                            adj_list[old_c][i].add(j)
                            adj_list[old_c][j].add(i)
                        found = True
                        break
                if found:
                    break
        
        return best_colors if best_c6s < current_c6s else None

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break

        found = False
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline - 15:
                break

            colors = _search(n, seed, deadline - 15)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    continue

        if found:
            break

    if best_n == 0:
        # Fallback to a known small valid coloring
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        
        if _verify(colors, n):
            best_n = n
            best_colors = colors
        else:
            n = 5
            best_n = n
            best_colors = [0] * (n*(n-1)//2)

    return {'n': best_n, 'colors': best_colors}
