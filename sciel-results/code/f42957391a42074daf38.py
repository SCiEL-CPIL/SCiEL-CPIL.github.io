import time
import sys

def construct():
    """
    Finds a dense triangle-free graph on 16 vertices that is not 3-colorable.
    Uses systematic backtracking with symmetry breaking and fast bitmask-based
    3-colorability checks to maximize edge count.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    # Bitmask for all vertices
    all_mask = (1 << n) - 1
    
    # Precompute bitmasks for all possible edges
    # edge_mask[i][j] is not needed, we just use (1<<u) | (1<<v) conceptually
    # But for adjacency, we use integer bitmasks.
    
    best_edge_count = 0
    best_graph_adj = [0] * n
    
    # Fast 3-colorability check using backtracking and bitmasks
    # adj: list of int bitmasks
    def is_3_colorable(adj):
        # We try to assign colors 0, 1, 2 to vertices.
        # To speed up, we use a recursive search with MRV (Minimum Remaining Values).
        # However, for n=16, a simple recursive check with static ordering 
        # might be fast enough if we prune early.
        # Let's use a slightly optimized recursive check.
        
        colors = [-1] * n
        # Order vertices by degree descending to fail early
        # Pre-compute degrees based on adj
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        
        def dfs(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Check available colors for v
            # A color c is invalid if any neighbor u has color c
            # Since we process in fixed order, some neighbors might be uncolored.
            # But wait, if we use fixed order, neighbors with higher index in 'order' 
            # are uncolored. This is a valid topological-like ordering for search, 
            # but not necessarily acyclic. 
            # Actually, standard backtracking on fixed order works:
            # When coloring v, only neighbors u where u is already colored (idx < current) 
            # restrict v? No, in graph coloring, constraints are symmetric.
            # If we fix an order, we only need to check neighbors that are ALREADY colored.
            # Neighbors not yet colored will be checked when they are colored.
            
            used_colors = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] & (1 << u)):
                    used_colors |= (1 << colors[u])
            
            available = 0b111 & ~used_colors
            if available == 0:
                return False
            
            # Try each available color
            # Heuristic: try color 0 first, etc.
            # To improve, we can try colors that are least used? 
            # For now, simple iteration.
            
            c = 0
            while available:
                if available & 1:
                    colors[v] = c
                    if dfs(idx + 1):
                        return True
                    colors[v] = -1
                available >>= 1
                c += 1
            
            return False
        
        return dfs(0)

    # Backtracking search to build the graph
    # We build the graph vertex by vertex.
    # adj: list of bitmasks
    # current_edge_count: int
    
    def backtrack(v, adj, edge_count):
        nonlocal best_edge_count, best_graph_adj
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: Even if we connect v to all remaining vertices, can we beat best?
        # Max possible edges added by v to vertices > v is (n - 1 - v)
        # But we also have edges among vertices > v not yet decided.
        # Upper bound: current edges + max possible future edges
        # Future vertices: v+1 to n-1
        # Max edges among them: (n-1-v)*(n-2-v)/2
        # Max edges from v to them: n-1-v
        # Total upper bound = edge_count + (n-1-v) + (n-1-v)*(n-2-v)//2
        # This is a loose bound. A tighter bound for triangle-free is Mantel's theorem on the subgraph?
        # Let's stick to a simple bound.
        
        if v == n:
            if edge_count > best_edge_count:
                # Check if 3-colorable (should be false if we pruned correctly, but verify)
                if not is_3_colorable(adj):
                    best_edge_count = edge_count
                    best_graph_adj = adj[:]
                    # print(f"New best: {best_edge_count} at v={v}")
            return
        
        # Try NOT adding any edges from v to previous vertices? 
        # No, we are adding edges from v to vertices 0..v-1.
        # Actually, it's better to decide edges (v, u) for u < v.
        # If we decide edges (v, u) for u < v, we ensure triangle freedom locally.
        
        # Let's reframe:
        # We decide the neighborhood of v among vertices {0, ..., v-1}.
        # Let S be the set of neighbors of v in {0, ..., v-1}.
        # For the graph to be triangle-free, S must be an independent set in the induced subgraph G[{0, ..., v-1}].
        # Why? If u, w in S and u-w is an edge, then (v, u, w) is a triangle.
        
        # So, we need to iterate over all independent sets of G[{0, ..., v-1}]?
        # That might be expensive.
        
        # Alternative: Iterate over subsets of {0, ..., v-1} that are independent.
        # We can generate them recursively or just check all 2^v subsets?
        # v goes up to 15. 2^15 = 32768. This is feasible per node if the tree isn't too deep.
        # But the tree depth is 16. 32768^16 is huge.
        # However, we prune heavily.
        
        # Let's optimize the independent set check.
        # We maintain adj as bitmasks.
        
        # Generate all subsets of {0, ..., v-1}
        # To do this efficiently, we can use a recursive helper or just loop 0 to 2^v - 1.
        # Given v is small on average, looping is okay.
        
        # But wait, we can be smarter.
        # We want to maximize edges. So we should try large independent sets first.
        
        # Let's collect all independent sets of the prefix graph.
        # Actually, we can just iterate through all masks from 0 to (1<<v)-1.
        # For each mask, check if it is an independent set.
        
        # Optimization:
        # If we pick a mask M for neighbors of v:
        # 1. M must be independent in G[0..v-1].
        # 2. We add edges (v, u) for u in M.
        # 3. We update adj.
        # 4. We recurse to v+1.
        
        # To maximize edges, we should try masks with high popcount first.
        
        # Let's precompute popcounts for all masks up to 2^15?
        # Or just use bit_count().
        
        # List of masks
        masks = list(range(1 << v))
        # Sort by popcount descending
        masks.sort(key=lambda m: -m.bit_count())
        
        for mask in masks:
            # Check if mask is an independent set in the current graph (vertices 0..v-1)
            is_independent = True
            # For each u in mask, check if it has any neighbor in mask
            # (adj[u] & mask) should be 0?
            # No, adj[u] contains neighbors in ALL vertices.
            # We need neighbors of u that are ALSO in mask.
            # So (adj[u] & mask) == 0 is correct for u in mask?
            # Yes, if u is in mask, and it has a neighbor w in mask, then (adj[u] & mask) != 0.
            
            # Wait, adj[u] includes v? No, we are building v. adj[u] only has edges to 0..u-1 (and potentially future? No, we build incrementally).
            # In our incremental construction, adj[u] only contains edges to vertices < u.
            # So when checking independence of mask (subset of 0..v-1), adj[u] only contains neighbors < u.
            # So (adj[u] & mask) works.
            
            # Optimization: If mask is 0, it's independent.
            if mask == 0:
                is_independent = True
            else:
                # Check each bit in mask
                m = mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    if adj[u] & mask:
                        is_independent = False
                        break
                    m -= lsb
                
            if not is_independent:
                continue
            
            # Check 3-colorability of the graph formed by 0..v with this neighborhood
            # We need to add edges (v, u) for u in mask to adj temporarily.
            
            new_adj = adj[:]
            # Add edges from v to mask
            # v is the new vertex.
            # new_adj[v] = mask
            # For each u in mask, new_adj[u] |= (1 << v)
            
            new_adj[v] = mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                new_adj[u] |= (1 << v)
                m -= lsb
            
            # Check 3-colorability
            # If it IS 3-colorable, we can prune this branch?
            # NO. If it IS 3-colorable, it might become non-3-colorable later?
            # Wait. The property "not 3-colorable" is monotonic with respect to adding edges.
            # If G is 3-colorable, any subgraph is 3-colorable.
            # If G is NOT 3-colorable, any supergraph is NOT 3-colorable.
            # We are ADDING edges.
            # So if the current graph (0..v) is 3-colorable, adding more edges (v+1..15) might make it non-3-colorable.
            # So we CANNOT prune if it is 3-colorable.
            
            # HOWEVER, if the current graph is NOT 3-colorable, then ANY extension is NOT 3-colorable.
            # So if it is NOT 3-colorable, we don't need to check 3-colorability for future extensions!
            # We can just greedily add maximum edges to the rest?
            # Not exactly, because we still need to maintain triangle-freeness.
            # But we don't need to check 3-colorability anymore.
            
            is_non_3c = not is_3_colorable(new_adj)
            
            # Recurse
            backtrack(v + 1, new_adj, edge_count + mask.bit_count(), is_non_3c)
            
            # If time is up, break
            if time.time() > deadline:
                return

    # We need to pass the is_non_3c flag to avoid re-checking.
    # Let's modify backtrack to accept this flag.
    
    def backtrack_optimized(v, adj, edge_count, already_non_3c):
        nonlocal best_edge_count, best_graph_adj
        
        if time.time() > deadline:
            return
        
        if v == n:
            if edge_count > best_edge_count:
                # If already_non_3c is True, it's valid.
                # If False, we must check. But if we didn't check in previous steps, we must check now.
                # Actually, if already_non_3c is False, it means every prefix was 3-colorable.
                # The final graph might be 3-colorable or not.
                if already_non_3c or not is_3_colorable(adj):
                    best_edge_count = edge_count
                    best_graph_adj = adj[:]
            return
        
        # Upper bound pruning
        # Max edges from v to future: n-1-v
        # Max edges among future: (n-1-v)*(n-2-v)//2
        # Total potential = edge_count + (n-1-v) + (n-1-v)*(n-2-v)//2
        # If this <= best_edge_count, prune.
        remaining_v = n - 1 - v
        max_future_edges = remaining_v + (remaining_v * (remaining_v - 1)) // 2
        if edge_count + max_future_edges <= best_edge_count:
            return
        
        # Generate independent sets for v
        # We want to try larger sets first to find dense graphs quickly.
        masks = list(range(1 << v))
        masks.sort(key=lambda m: -m.bit_count())
        
        for mask in masks:
            # Check independence
            if mask == 0:
                is_independent = True
            else:
                is_independent = True
                m = mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    if adj[u] & mask:
                        is_independent = False
                        break
                    m -= lsb
            
            if not is_independent:
                continue
            
            # Update adj
            new_adj = adj[:]
            new_adj[v] = mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                new_adj[u] |= (1 << v)
                m -= lsb
            
            new_edge_count = edge_count + mask.bit_count()
            
            # Check 3-colorability only if we haven't already established non-3-colorability
            if not already_non_3c:
                # Check if this step makes it non-3-colorable
                if not is_3_colorable(new_adj):
                    already_non_3c = True
                # If it is still 3-colorable, we continue with already_non_3c = False
            
            backtrack_optimized(v + 1, new_adj, new_edge_count, already_non_3c)
            
            if time.time() > deadline:
                return

    # Start search
    # Initial state: v=0, no edges, adj all 0
    initial_adj = [0] * n
    backtrack_optimized(0, initial_adj, 0, False)
    
    # Convert best_graph_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_graph_adj[u] & (1 << v):
                result.append([u, v])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
