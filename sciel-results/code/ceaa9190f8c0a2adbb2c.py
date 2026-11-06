"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    if adj[i] & adj[j]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using fast backtracking with dynamic variable ordering.
        Returns True if 3-colorable, False if not (i.e., 4-chromatic or higher).
        """
        # Precompute neighbor masks
        nmask = [adj[i] for i in range(16)]
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        
        # Use dynamic ordering: always pick the uncolored vertex with most constraints
        # This is faster for dense graphs
        
        uncolored_mask = (1 << 16) - 1
        color_masks = [0, 0, 0]
        
        def backtrack(uncolored, depth):
            if uncolored == 0:
                return True
            
            # Pick vertex with most colored neighbors (most constrained)
            best_v = -1
            best_score = -1
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Score: number of colored neighbors
                score = bin(nmask[v] & (color_masks[0] | color_masks[1] | color_masks[2])).count('1')
                # Tie-break by degree
                score = score * 100 + degrees[v]
                if score > best_score:
                    best_score = score
                    best_v = v
            
            v = best_v
            vmask = 1 << v
            vn = nmask[v]
            
            # Determine available colors
            forbidden = 0
            if vn & color_masks[0]:
                forbidden |= 1
            if vn & color_masks[1]:
                forbidden |= 2
            if vn & color_masks[2]:
                forbidden |= 4
            
            available = 7 & (~forbidden)
            
            if available == 0:
                return False
            
            # Try each available color
            while available:
                lsb = available & (-available)
                c = 0 if lsb == 1 else (1 if lsb == 2 else 2)
                available ^= lsb
                
                color_masks[c] |= vmask
                
                if backtrack(uncolored ^ vmask, depth + 1):
                    return True
                
                color_masks[c] ^= vmask
            
            return False
        
        return backtrack((1 << 16) - 1, 0)
    
    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2
    
    def find_critical_core(adj):
        """Find a minimal 4-chromatic subgraph (critical core) of the given graph.
        Returns a bitmask of vertices in the core, or None if graph is 3-colorable.
        """
        if is_3_colorable(adj):
            return None
        
        # Try to find a small critical subgraph by removing vertices one at a time
        core_mask = 0
        for i in range(16):
            core_mask |= (1 << i)
        
        # Greedily try to remove vertices while maintaining 4-chromaticity
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(16):
                if not (core_mask & (1 << i)):
                    continue
                # Try removing vertex i
                new_mask = core_mask & ~(1 << i)
                if new_mask == 0:
                    continue
                
                # Build subgraph on new_mask
                sub_adj = [0] * 16
                for u in range(16):
                    if not (new_mask & (1 << u)):
                        continue
                    for v in range(u + 1, 16):
                        if (new_mask & (1 << v)) and (adj[u] & (1 << v)):
                            sub_adj[u] |= (1 << v)
                            sub_adj[v] |= (1 << u)
                
                # Check if subgraph is still 4-chromatic
                if not is_3_colorable(sub_adj):
                    core_mask = new_mask
                    changed = True
                    break
        
        return core_mask
    
    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * 16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def build_grotzsch():
        """Build the Grötzsch graph (Mycielski of C5) on vertices 0-10."""
        adj = [0] * 16
        # C5 on vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        return adj
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        if common:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v). Returns True if removed."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    def systematic_maximize(adj, deadline):
        """Systematically try to add all possible edges, maintaining 4-chromaticity.
        Uses core-constrained search: prioritize edges that preserve the critical core.
        """
        current = adj[:]
        current_count = count_edges(current)
        
        # Find the critical core
        core_mask = find_critical_core(current)
        
        # Get all possible edges not in current
        possible = []
        for i in range(16):
            for j in range(i + 1, 16):
                if not (current[i] & (1 << j)):
                    possible.append((i, j))
        
        # Sort edges: prioritize those involving core vertices, then by degree sum
        def edge_priority(e):
            u, v = e
            # Higher priority for edges touching core vertices
            core_u = 1 if (core_mask and (core_mask & (1 << u))) else 0
            core_v = 1 if (core_mask and (core_mask & (1 << v))) else 0
            degree_sum = bin(current[u]).count('1') + bin(current[v]).count('1')
            # Edges within core first, then core-noncore, then noncore-noncore
            return (core_u + core_v, degree_sum, u, v)
        
        possible.sort(key=edge_priority, reverse=True)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in possible:
                if time.time() > deadline:
                    break
                # Check if edge still possible
                if current[u] & (1 << v):
                    continue
                common = current[u] & current[v]
                if common:
                    continue
                
                # Quick check: if edge would complete a triangle in the core, skip
                if core_mask:
                    # Check if there's a common neighbor that's also in the core
                    common_core = common & core_mask
                    if common_core:
                        # This edge would create a triangle with a core vertex - but we need to check
                        # if it actually creates a triangle (common neighbor must be adjacent to both u and v)
                        # common already ensures this, so skip if common_core is non-empty
                        # Actually, common = adj[u] & adj[v], so any w in common is adjacent to both u and v
                        # So adding (u,v) with w in common creates triangle u-v-w
                        # But we already checked common == 0 above, so this can't happen
                        pass
                
                # Try adding
                current[u] |= (1 << v)
                current[v] |= (1 << u)
                
                if not is_3_colorable(current):
                    current_count += 1
                    improved = True
                else:
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
        
        return current, current_count
    
    def random_triangle_free_graph():
        """Generate a random triangle-free graph with bias toward denser graphs."""
        adj = [0] * 16
        all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            common = adj[u] & adj[v]
            if common == 0:
                # Accept with probability that increases with current density
                current_density = count_edges(adj) / 120.0
                prob = 0.3 + 0.5 * current_density
                if random.random() < prob:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Start from Grötzsch graph and maximize
    grotzsch = build_grotzsch()
    if is_triangle_free(grotzsch) and not is_3_colorable(grotzsch):
        maxed, cnt = systematic_maximize(grotzsch, deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = maxed[:]
    
    # Phase 2: Try random triangle-free graphs, biased toward dense ones
    # Use the faster is_3_colorable to allow more restarts
    for _ in range(400):
        if time.time() > deadline - 20:
            break
        adj = random_triangle_free_graph()
        cnt = count_edges(adj)
        if cnt >= best_count - 5 and not is_3_colorable(adj):
            maxed, new_cnt = systematic_maximize(adj, deadline)
            if new_cnt > best_count:
                best_count = new_cnt
                best_adj = maxed[:]
    
    # Phase 3: If we still have time, try more random starts with lower threshold
    if time.time() < deadline - 25:
        for _ in range(500):
            if time.time() > deadline - 20:
                break
            adj = random_triangle_free_graph()
            cnt = count_edges(adj)
            if cnt >= best_count - 10 and not is_3_colorable(adj):
                maxed, new_cnt = systematic_maximize(adj, deadline)
                if new_cnt > best_count:
                    best_count = new_cnt
                    best_adj = maxed[:]
    
    # Phase 4: Final local search from best with core-aware ordering
    # Multiple passes to find more improvements
    if best_adj is not None and time.time() < deadline - 10:
        current = best_adj[:]
        current_count = best_count
        core_mask = find_critical_core(current)
        
        for _pass in range(3):
            if time.time() > deadline - 5:
                break
            
            # Try all single-edge additions, prioritizing core-touching edges
            candidates = []
            for i in range(16):
                for j in range(i + 1, 16):
                    if current[i] & (1 << j):
                        continue
                    common = current[i] & current[j]
                    if common:
                        continue
                    core_i = 1 if (core_mask and (core_mask & (1 << i))) else 0
                    core_j = 1 if (core_mask and (core_mask & (1 << j))) else 0
                    candidates.append((core_i + core_j, i, j))
            
            candidates.sort(reverse=True)
            
            improved_this_pass = False
            for _, i, j in candidates:
                if time.time() > deadline - 5:
                    break
                current[i] |= (1 << j)
                current[j] |= (1 << i)
                
                if not is_3_colorable(current):
                    current_count += 1
                    best_count = current_count
                    best_adj = current[:]
                    improved_this_pass = True
                else:
                    current[i] &= ~(1 << j)
                    current[j] &= ~(1 << i)
            
            if not improved_this_pass:
                break
    
    if best_adj is None:
        # Absolute fallback: return the Grötzsch graph
        grotzsch = build_grotzsch()
        return adj_to_edges(grotzsch)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
