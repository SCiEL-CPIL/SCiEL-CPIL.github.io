import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts_bitmask(edges, n):
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        n_edges = len(edges)
        conflicts = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            near_vertices = set(adj[u]) | set(adj[v])
            near_vertices.add(u)
            near_vertices.add(v)
            
            for x in near_vertices:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)
                            
        return conflicts

    def greedy_color_fast(order, conflicts, n_edges):
        """Fast greedy coloring using bitmask conflict checks."""
        color = [-1] * n_edges
        color_masks = []  # color_masks[c] = bitmask of edges with color c
        
        for idx in order:
            c = 0
            while c < len(color_masks) and (conflicts[idx] & color_masks[c]):
                c += 1
            
            color[idx] = c
            if c >= len(color_masks):
                color_masks.append(0)
            color_masks[c] |= (1 << idx)
            
        return color, len(color_masks)

    def local_polish_fast(color, conflicts, n_edges, deadline):
        """
        Fast local search using bitmask conflict checks.
        """
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
            
        # Build color class bitmasks
        class_masks = [0] * n_colors
        for i, c in enumerate(color):
            class_masks[c] |= (1 << i)
            
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # 1. Try to eliminate colors (from highest to lowest)
            for c in range(n_colors - 1, -1, -1):
                if not class_masks[c]:
                    continue
                temp_assign = {}
                success = True
                for e in range(n_edges):
                    if not (class_masks[c] >> e) & 1:
                        continue
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not (conflicts[e] & class_masks[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    for e, c2 in temp_assign.items():
                        class_masks[c] &= ~(1 << e)
                        class_masks[c2] |= (1 << e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
            
            # 2. Move individual edges to lower colors
            if not improved:
                for c in range(n_colors - 1, -1, -1):
                    if not class_masks[c]:
                        continue
                    for e in range(n_edges):
                        if not (class_masks[c] >> e) & 1:
                            continue
                        for c2 in range(c):
                            if not (conflicts[e] & class_masks[c2]):
                                class_masks[c] &= ~(1 << e)
                                class_masks[c2] |= (1 << e)
                                color[e] = c2
                                improved = True
                                break
                        if improved:
                            if not class_masks[n_colors - 1]:
                                n_colors -= 1
                            break
                    if improved:
                        break
                        
        return color, n_colors

    def max_clique_bnb(adj, n, time_limit=1.0):
        """
        Find Maximum Clique in graph defined by adj (list of bitmasks).
        adj[i] is bitmask of neighbors of i.
        Returns list of indices in the clique.
        """
        best_clique = []
        best_size = 0
        start = time.time()
        
        def greedy_color_bound(cand):
            colors = 0
            P = cand
            while P:
                colors += 1
                Q = P
                while Q:
                    v_bit = Q & (-Q)
                    v = v_bit.bit_length() - 1
                    P &= ~v_bit
                    Q &= ~v_bit
                    Q &= ~adj[v]
            return colors
        
        def expand(candidates, current_clique):
            nonlocal best_clique, best_size
            
            if time.time() - start > time_limit:
                return
            
            bound = greedy_color_bound(candidates)
            if len(current_clique) + bound <= best_size:
                return
            
            if not candidates:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = current_clique[:]
                return
            
            # Pick vertex with max degree in candidates for better pruning
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            best_v = v
            best_deg = (candidates & adj[v]).bit_count()
            temp = candidates
            while temp:
                tb = temp & (-temp)
                tv = tb.bit_length() - 1
                deg = (candidates & adj[tv]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = tv
                temp &= ~tb
            v = best_v
            v_bit = 1 << v
            
            # Branch 1: Include v
            new_candidates = candidates & adj[v] & ~v_bit
            current_clique.append(v)
            expand(new_candidates, current_clique)
            current_clique.pop()
            
            # Branch 2: Exclude v
            new_candidates = candidates & ~v_bit
            expand(new_candidates, current_clique)

        expand((1 << n) - 1, [])
        
        return best_clique

    def construct_imim(edges, conflicts, n_edges, deadline):
        """
        Iterative Maximum Induced Matching with B&B and proper fallback.
        """
        color = [-1] * n_edges
        remaining = (1 << n_edges) - 1
        current_color = 0
        
        while remaining:
            # Time check: if running out of time, use greedy for remaining
            if time.time() > deadline - 8.0:
                rem_indices = [i for i in range(n_edges) if (remaining >> i) & 1]
                # Greedy color the remaining edges
                order = rem_indices
                sub_color, _ = greedy_color_fast(order, conflicts, n_edges)
                for i in rem_indices:
                    color[i] = current_color + sub_color[i]
                current_color += len(set(sub_color[i] for i in rem_indices))
                break
            
            rem_indices = [i for i in range(n_edges) if (remaining >> i) & 1]
            if not rem_indices:
                break
                
            k = len(rem_indices)
            if k == 1:
                color[rem_indices[0]] = current_color
                remaining &= ~(1 << rem_indices[0])
                current_color += 1
                continue
            
            # Build adj for the subgraph (Complement of Conflict)
            sub_adj = [0] * k
            for i_idx, i_global in enumerate(rem_indices):
                mask = remaining & ~conflicts[i_global]
                local_mask = 0
                for j_idx, j_global in enumerate(rem_indices):
                    if (mask >> j_global) & 1:
                        local_mask |= (1 << j_idx)
                sub_adj[i_idx] = local_mask
            
            # Run B&B with adaptive time limit
            time_left = deadline - time.time() - 5.0
            bnb_time = min(2.0, max(0.1, time_left / max(1, n_edges - len(rem_indices) + 1)))
            clique = max_clique_bnb(sub_adj, k, time_limit=bnb_time)
            
            if not clique:
                clique = [0]
            
            # Assign color
            for i_idx in clique:
                i_global = rem_indices[i_idx]
                color[i_global] = current_color
                remaining &= ~(1 << i_global)
            
            current_color += 1
            
        return color, current_color

    def multi_restart_greedy(edges, conflicts, n_edges, deadline, n_restarts=5):
        """
        Multi-restart greedy with random ordering to find good initial solutions.
        """
        best_color = None
        best_n_colors = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random ordering
            order = list(range(n_edges))
            random.shuffle(order)
            
            color, n_colors = greedy_color_fast(order, conflicts, n_edges)
            
            if n_colors < best_n_colors:
                best_n_colors = n_colors
                best_color = color[:]
        
        if best_color is None:
            order = list(range(n_edges))
            best_color, best_n_colors = greedy_color_fast(order, conflicts, n_edges)
        
        return best_color, best_n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        n_edges = len(edges)
        conflicts = build_conflicts_bitmask(edges, n)
        
        # Strategy 1: Multi-restart greedy
        color_greedy, n_colors_greedy = multi_restart_greedy(edges, conflicts, n_edges, deadline, n_restarts=3)
        
        # Strategy 2: IMIM construction
        color_imim, n_colors_imim = construct_imim(edges, conflicts, n_edges, deadline)
        
        # Pick the better initial solution
        if n_colors_imim < n_colors_greedy:
            color, n_colors = color_imim, n_colors_imim
        else:
            color, n_colors = color_greedy, n_colors_greedy
        
        # Local polish
        color, n_colors = local_polish_fast(color, conflicts, n_edges, deadline)
        
        results.append(color)
        
    return results
