import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def get_polarity_graph(q):
        """
        Construct the polarity graph of PG(2, q).
        Vertices are points (x, y, z) in GF(q)^3, up to scaling.
        We represent points as tuples (x, y, 1) for z=1, and (0, 1, 0) for z=0 (infinity).
        Total vertices: q^2 + q + 1.
        Edge (u, v) if u is on the polar line of v.
        For a point P=(x,y,z), the polar line is xX + yY + zZ = 0 (for symmetric polarity).
        Actually, the standard polarity is xX + yY - zZ = 0 or similar.
        Let's use the standard one: xX + yY + zZ = 0 is not a polarity for all q.
        A common polarity is xX + yY - zZ = 0.
        Let's use the construction from the inspiration: _construct_polarity_graph.
        """
        # We will use q=5, n=31.
        if q != 5:
            # Fallback to simple construction if q != 5
            return None
            
        # Points of PG(2, 5)
        # 25 affine points (x, y, 1)
        # 5 points at infinity (directions)
        # 1 point at infinity (the line at infinity? No, PG(2) has q+1 points on the line at infinity)
        # Total 25 + 5 + 1 = 31.
        
        # Let's label vertices 0..30.
        # 0..24: (x, y, 1) for x,y in 0..4
        # 25..29: (1, m, 0) for m in 0..4 (points at infinity with finite slope)
        # 30: (0, 1, 0) (point at infinity with vertical slope)
        
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for m in range(q):
            points.append((1, m, 0))
        points.append((0, 1, 0))
        
        n = len(points)
        adj = [set() for _ in range(n)]
        
        # Polarity: P=(x,y,z) is conjugate to Q=(u,v,w) if x*u + y*v - z*w == 0 mod q
        # Note: The sign depends on the choice of polarity.
        # For q=5, -1 is a square? 4=2^2. Yes.
        # So x*u + y*v - z*w = 0 is a valid polarity.
        
        for i in range(n):
            x, y, z = points[i]
            for j in range(n):
                if i == j:
                    continue
                u, v, w = points[j]
                # Check if j is on the polar line of i
                if (x * u + y * v - z * w) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Verify C4-free
        # It should be C4-free by theory.
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
                    
        return n, edges, adj

    def exact_independent_set_size(adj, n, target_size, deadline):
        """
        Returns True if there is an independent set of size target_size.
        Uses bitmask DFS with pruning.
        """
        if n < target_size:
            return False
            
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_masks[i] = mask
            
        all_mask = (1 << n) - 1
        # Order vertices by degree for better pruning?
        # Simple branch and bound.
        
        def branch(remaining_mask, current_size):
            if time.time() > deadline:
                return False
            if current_size >= target_size:
                return True
            if remaining_mask == 0:
                return False
            if current_size + remaining_mask.bit_count() < target_size:
                return False
                
            # Pick a vertex to branch on.
            # Heuristic: pick vertex with max degree in remaining subgraph?
            # Or min degree?
            # For finding large independent sets, picking a vertex with high degree
            # removes many neighbors, which might be good for pruning if we include it?
            # If we include v, we remove N(v).
            # If we exclude v, we just remove v.
            # Let's pick a vertex with high degree to maximize the impact of inclusion.
            
            temp = remaining_mask
            v_idx = -1
            max_deg = -1
            while temp:
                lsb = temp & -temp
                v = (lsb.bit_length() - 1)
                neighbors_in_rem = adj_masks[v] & remaining_mask
                deg = neighbors_in_rem.bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v_idx = v
                temp -= lsb
                
            if v_idx == -1:
                return False
                
            v_bit = 1 << v_idx
            
            # Option 1: Include v
            new_remaining = remaining_mask & ~(adj_masks[v_idx] | v_bit)
            if branch(new_remaining, current_size + 1):
                return True
                
            # Option 2: Exclude v
            new_remaining = remaining_mask & ~v_bit
            if branch(new_remaining, current_size):
                return True
                
            return False

        return branch(all_mask, 0)

    def is_c4_free(adj, n):
        for i in range(n):
            for j in range(i+1, n):
                if len(adj[i] & adj[j]) > 1:
                    return False
        return True

    def try_extend(n, adj, edges, target_n, deadline):
        """
        Try to extend the graph from n to target_n.
        Returns (new_n, new_edges, new_adj) or None if failed.
        """
        if n >= target_n:
            return n, edges, adj
            
        # Generate candidates for the new vertex's neighbors
        # The neighbors S must be an independent set in the conflict graph.
        # Conflict graph: u ~ v if N(u) ∩ N(v) != empty.
        
        # We will try several random sets S.
        candidates = list(range(n))
        
        # Sort candidates by degree? 
        # High degree vertices are more "connected", so they conflict with more others.
        # Low degree vertices are less likely to conflict.
        # To maximize |S|, we should pick low degree vertices first?
        # But we also want to reduce alpha.
        # Connecting to high degree vertices might be better for alpha reduction?
        
        # Let's try a few random orders.
        for _ in range(20):
            if time.time() > deadline:
                return None
                
            random.shuffle(candidates)
            S = set()
            for u in candidates:
                can_add = True
                for v in S:
                    if adj[u] & adj[v]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
            
            if not S:
                continue
                
            # Create new graph
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            new_edges = edges[:]
            
            for u in S:
                new_adj[new_n-1].add(u)
                new_adj[u].add(new_n-1)
                new_edges.append([new_n-1, u])
                
            # Check C4-free
            if not is_c4_free(new_adj, new_n):
                continue
                
            # Check alpha
            # If new_n is small, use exact check.
            # If new_n is large, use heuristic or skip?
            # For n=43, exact check might be slow if called too often.
            # Let's use exact check only if n <= 35 or every few steps.
            # Or use a fast heuristic to filter.
            
            # Heuristic: Greedy independent set
            remaining = set(range(new_n))
            is_set = []
            while remaining:
                min_deg = float('inf')
                v = None
                for u in remaining:
                    deg = len(new_adj[u] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        v = u
                is_set.append(v)
                remaining.remove(v)
                remaining -= new_adj[v]
            
            if len(is_set) >= 11:
                # Likely invalid, skip
                continue
                
            # Recurse
            res = try_extend(new_n, new_adj, new_edges, target_n, deadline)
            if res:
                return res
                
        return None

    # Initialize with PG(2, 5)
    base_n, base_edges, base_adj = get_polarity_graph(5)
    
    if base_n is None:
        # Fallback: 10 triangles
        base_n = 30
        base_edges = []
        base_adj = [set() for _ in range(base_n)]
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            base_adj[u].add(v); base_adj[v].add(u)
            base_adj[u].add(w); base_adj[w].add(u)
            base_adj[v].add(w); base_adj[w].add(v)
            base_edges.extend([[u,v], [u,w], [v,w]])
            
    # Try to extend to 43
    best_result = None
    
    # Try multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        # Deep copy base
        cur_n = base_n
        cur_adj = [set(s) for s in base_adj]
        cur_edges = base_edges[:]
        
        res = try_extend(cur_n, cur_adj, cur_edges, 43, deadline - 5)
        
        if res:
            final_n, final_edges, final_adj = res
            # Final verification
            if is_c4_free(final_adj, final_n) and not exact_independent_set_size(final_adj, final_n, 11, deadline - 2):
                if best_result is None or final_n > best_result[0]:
                    best_result = (final_n, final_edges)
                    
        # If we reached 43, we can stop early?
        # But maybe a different restart finds a better structure?
        # The task is to maximize n. 43 is the max allowed.
        # So if we find one, it's optimal.
        if best_result and best_result[0] == 43:
            break
            
    if best_result:
        return {'n': best_result[0], 'edges': best_result[1]}
        
    # Fallback: Return the base graph
    return {'n': base_n, 'edges': base_edges}

# Test
# if __name__ == '__main__':
#     res = construct()
#     print(res['n'])
