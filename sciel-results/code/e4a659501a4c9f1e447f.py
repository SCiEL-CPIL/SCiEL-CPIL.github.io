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
                # Normalize: ensure smallest vertex is first? 
                # Since 0 is fixed, we just need to handle reflection.
                # Reverse cycle: 0, 5, 4, 3, 2, 1
                rev_cycle = (0,) + tuple(reversed(perm))
                # Canonical form: min(cycle, rev_cycle) lexicographically?
                # Actually, just store the edge set to avoid duplicates in verification.
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

    def _precompute_path4(v, adj_list):
        """
        Compute path4_mask[c][i] for subgraph on vertices 0..v-1.
        path4_mask[c][i] is a bitmask where bit d is set if there is a 
        simple path of length 4 from i to d in color c graph.
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                neighbors_i = adj_list[c][i]
                if not neighbors_i:
                    continue
                
                mask = 0
                # Path: i -> a -> b -> e -> d
                for a in neighbors_i:
                    # b must be neighbor of a, not i
                    for b in adj_list[c][a]:
                        if b == i:
                            continue
                        # e must be neighbor of b, not i, a
                        for e in adj_list[c][b]:
                            if e == i or e == a:
                                continue
                            # d must be neighbor of e, not i, a, b
                            for d in adj_list[c][e]:
                                if d == i or d == a or d == b:
                                    continue
                                mask |= (1 << d)
                path4_mask[c][i] = mask
        
        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        
        # Shuffle color order for this search attempt
        color_order = [0, 1, 2, 3]
        rng.shuffle(color_order)

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            # This is the most expensive part per vertex
            path4_mask = _precompute_path4(v, adj_list)

            # Backtrack over edges (v, 0), (v, 1), ..., (v, v-1)
            # masks[c] holds bits for vertices i < current_idx assigned color c to (v, i)
            masks = [0, 0, 0, 0]
            
            # Shuffle the order in which we assign edges to neighbors
            neighbors_order = list(range(v))
            rng.shuffle(neighbors_order)
            
            def _backtrack(pos):
                if time.time() > deadline:
                    return False

                if pos == v:
                    return True

                target_vertex = neighbors_order[pos]
                
                # Try each color for edge (v, target_vertex)
                # Heuristic: try colors in shuffled order
                for c in color_order:
                    # Check constraint: 
                    # If we assign color c to (v, target_vertex), then target_vertex must not be 
                    # connected via Path4 to any vertex j already assigned color c.
                    if path4_mask[c][target_vertex] & masks[c]:
                        continue
                    
                    # Assign
                    masks[c] |= (1 << target_vertex)
                    adj[target_vertex][v] = c
                    adj[v][target_vertex] = c
                    adj_list[c][target_vertex].add(v)
                    adj_list[c][v].add(target_vertex)

                    if _backtrack(pos + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << target_vertex)
                    adj[target_vertex][v] = -1
                    adj[v][target_vertex] = -1
                    adj_list[c][target_vertex].remove(v)
                    adj_list[c][v].remove(target_vertex)

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

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    # 17 is the target. If we find it, we are done.
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
                    # Should not happen if logic is correct, but safety
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
            # Extremely unlikely, but return a trivial valid one
            n = 5
            best_n = n
            best_colors = [0] * (n*(n-1)//2)

    return {'n': best_n, 'colors': best_colors}
