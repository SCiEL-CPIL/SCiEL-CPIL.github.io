"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
            # Neighbors of u and v
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
                            
        # Build conflict lists for faster iteration
        conflict_list = [[] for _ in range(n_edges)]
        for i in range(n_edges):
            mask = conflicts[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                conflict_list[i].append(j)
                mask ^= lsb
        return conflicts, conflict_list

    def greedy_color(order, conflicts, n_edges):
        """Standard greedy coloring for initialization or fallback."""
        color = [-1] * n_edges
        color_to_edges = []
        
        for idx in order:
            used_mask = 0
            for c_idx, edges_in_color in enumerate(color_to_edges):
                # Check if idx conflicts with any edge in color c_idx
                # We can precompute a mask for each color class to speed this up
                # But for simplicity in this helper, we check directly
                if any(conflicts[idx] & (1 << e) for e in edges_in_color):
                    used_mask |= (1 << c_idx)
            
            c = 0
            while (used_mask >> c) & 1:
                c += 1
            
            color[idx] = c
            if c >= len(color_to_edges):
                color_to_edges.append(set())
            color_to_edges[c].add(idx)
            
        return color, len(color_to_edges)

    def max_induced_matching_greedy(conflicts, conflict_list, n_edges, remaining_mask, time_limit=0.5):
        """
        Find a large induced matching in the subgraph induced by remaining_mask.
        Uses a greedy approach with randomization and local optimization.
        Returns a list of edge indices forming the matching.
        """
        start = time.time()
        best_matching = []
        best_size = 0
        
        # Multiple restarts
        for restart in range(10):
            if time.time() - start > time_limit:
                break
            
            matching = []
            available = remaining_mask
            
            # Order edges by degree in the available subgraph (ascending)
            # To speed this up, we'll just iterate through available bits
            # A better heuristic: pick edge with fewest conflicts in available set
            # For now, simple greedy with random tie-breaking
            
            # Create a list of available edges
            avail_edges = []
            m = available
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                avail_edges.append(idx)
                m ^= lsb
            
            # Shuffle for randomization
            random.shuffle(avail_edges)
            
            # Greedy selection
            current_matching = []
            current_mask = remaining_mask
            
            for e in avail_edges:
                if not (current_mask & (1 << e)):
                    continue
                
                # Check if e can be added: it must not conflict with any edge in current_matching
                # And it must not have been removed
                can_add = True
                for e2 in current_matching:
                    if conflicts[e] & (1 << e2):
                        can_add = False
                        break
                
                if can_add:
                    current_matching.append(e)
                    # Remove e and all its neighbors from current_mask
                    # Neighbors in the conflict graph
                    to_remove = (1 << e) | conflicts[e]
                    current_mask &= ~to_remove
            
            if len(current_matching) > best_size:
                best_size = len(current_matching)
                best_matching = current_matching[:]
                
            # Try to improve best_matching with local search (swap)
            # This is complex, so we rely on multiple restarts
            
        return best_matching

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        conflicts, conflict_list = build_conflicts_bitmask(edges, n)
        
        # Strategy: Iteratively find large induced matchings
        # 1. Initialize with greedy DSATUR or similar to get a baseline
        # 2. Use iterative induced matching extraction to refine
        
        # First, get a good initial coloring using DSATUR-like greedy
        # Order by degree
        order = list(range(m))
        order.sort(key=lambda i: len(conflict_list[i]), reverse=True)
        
        color, nc = greedy_color(order, conflicts, m)
        
        # Now, try to improve by extracting large induced matchings
        # We will try to rebuild the coloring by picking large induced matchings
        # from the remaining graph
        
        # We'll do a few rounds of "extract matching, assign color, repeat"
        # But we want to minimize total colors, so we want each matching to be as large as possible
        
        # Let's try a different approach: 
        # Keep the initial coloring, but try to merge color classes or replace them with larger induced matchings
        
        # Actually, the "iterative maximum induced matching" strategy implies:
        # 1. Find Max Induced Matching M1 in G
        # 2. Remove M1 and its neighbors from G
        # 3. Find Max Induced Matching M2 in G'
        # 4. Repeat
        # This yields a coloring where each color class is a maximal induced matching.
        # This is generally better than greedy DSATUR because it optimizes each class size.
        
        # Let's implement this directly.
        
        best_color = color[:]
        best_nc = nc
        
        # Try the iterative extraction method
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            current_color = [-1] * m
            remaining_mask = (1 << m) - 1
            current_nc = 0
            
            # We need to ensure we don't exceed the time budget significantly
            # The max_induced_matching_greedy is fast enough for small m
            
            while remaining_mask:
                if time.time() > deadline - 10:
                    break
                    
                # Find a large induced matching in the remaining graph
                # We use a simple greedy with randomization
                # To make it "maximum" like, we use multiple restarts inside
                matching = max_induced_matching_greedy(conflicts, conflict_list, m, remaining_mask, time_limit=0.2)
                
                if not matching:
                    # Should not happen if remaining_mask is not 0, but just in case
                    # Assign a new color to the first available edge
                    lsb = remaining_mask & -remaining_mask
                    idx = lsb.bit_length() - 1
                    current_color[idx] = current_nc
                    remaining_mask &= ~(1 << idx)
                    current_nc += 1
                    continue
                
                # Assign current_nc to all edges in matching
                for e in matching:
                    current_color[e] = current_nc
                
                # Remove matching and its neighbors from remaining_mask
                remove_mask = 0
                for e in matching:
                    remove_mask |= (1 << e)
                    remove_mask |= conflicts[e]
                
                remaining_mask &= ~remove_mask
                current_nc += 1
            
            if current_nc < best_nc:
                best_nc = current_nc
                best_color = current_color[:]
        
        # Polish the best color found
        # Try to reduce colors further by moving edges
        # Simple pass: for each color from high to low, try to move edges to lower colors
        for c in range(best_nc - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            edges_in_color = [i for i in range(m) if best_color[i] == c]
            for e in edges_in_color:
                # Try to move e to a lower color
                moved = False
                for c2 in range(c):
                    # Check if e conflicts with any edge in color c2
                    can_move = True
                    for e2 in range(m):
                        if best_color[e2] == c2 and (conflicts[e] & (1 << e2)):
                            can_move = False
                            break
                    if can_move:
                        best_color[e] = c2
                        moved = True
                        break
                if moved:
                    # Check if color c is now empty
                    if not any(best_color[i] == c for i in range(m)):
                        best_nc -= 1
        
        return best_color

    # Allocate time budget per instance
    total_budget = deadline - start_time - 5
    weights = [40*4, 60*5, 80*6]
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        n, d, seed = spec
        remaining = deadline - time.time() - 5
        budget = min(budgets[i], remaining)
        if budget < 5:
            out.append([])
            continue
        out.append(solve_instance(n, d, seed, budget))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
