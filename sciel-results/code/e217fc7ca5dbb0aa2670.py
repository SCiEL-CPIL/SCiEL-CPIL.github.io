import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_valid_after_add(adj, u, v):
        """
        Fast check: after adding edge (u,v), check if K3,3 is created.
        Only need to check triples where the new edge adds a common neighbor.
        For triple {u,a,b}: new common neighbor is v iff v in N(a) and v in N(b).
          Invalid if |N(u)∩N(a)∩N(b) minus {v}| == 2 and v in N(a)∩N(b).
        For triple {v,a,b}: new common neighbor is u iff u in N(a) and u in N(b).
          Invalid if |N(v)∩N(a)∩N(b) minus {u}| == 2 and u in N(a)∩N(b).
        """
        # For triples {u, a, b}: v is the new common neighbor if v in N(a) and v in N(b)
        # Check if old common count (without v) is exactly 2
        vu = adj[u] & ~(1 << v)  # N(u) without v
        for a in range(n):
            if a == u:
                continue
            if not (adj[a] & (1 << v)):  # v not in N(a), skip
                continue
            for b in range(a + 1, n):
                if b == u:
                    continue
                if not (adj[b] & (1 << v)):  # v not in N(b), skip
                    continue
                # v is in N(a)∩N(b), so it's a new common neighbor for {u,a,b}
                # Check if old common count (without v) is 2
                old_common = vu & adj[a] & adj[b]
                if old_common.bit_count() == 2:
                    return False
        
        # For triples {v, a, b}: u is the new common neighbor if u in N(a) and u in N(b)
        vv = adj[v] & ~(1 << u)  # N(v) without u
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):  # u not in N(a), skip
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):  # u not in N(b), skip
                    continue
                # u is in N(a)∩N(b), so it's a new common neighbor for {v,a,b}
                # Check if old common count (without u) is 2
                old_common = vv & adj[a] & adj[b]
                if old_common.bit_count() == 2:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid_after_add(adj, u, v):
                pass # Keep edge
            else:
                # Rollback
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    def make_grid_start():
        """
        Create a 5x4 grid-like bipartite structure as a strong starting point.
        Partition vertices into 5 groups of 4. Connect groups in a pattern
        that avoids K3,3. Use a circulant-like connection pattern.
        """
        adj = [0] * n
        # Groups: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}, G4={16,17,18,19}
        # Connect group i to group j if (j-i) mod 5 is in {1,2} (each group connects to 2 neighbors)
        # Within each connection, connect all pairs (complete bipartite between groups)
        # This gives 5 * 2 * 4 * 4 / 2 = 80 edges... but need to check K3,3 freeness
        # Actually let's be more careful. Use a 5-cycle of groups, each group connects to 
        # its two neighbors on the cycle. Within each inter-group connection, use a 
        # 4x4 complete bipartite.
        # But complete bipartite K4,4 between adjacent groups might create K3,3.
        # Let's use a sparser pattern: each vertex in group i connects to 
        # specific vertices in groups i+1 and i-1 mod 5.
        
        # Simpler: start with a random graph and let greedy fill do the work,
        # but seed with some structure. Let's just use a random initial graph.
        adj = [0] * n
        return adj

    best_adj = [0] * n
    best_count = 0
    
    # Try multiple random restarts with greedy fill
    for restart in range(50):
        if time.time() > deadline:
            break
        current_adj = [0] * n
        current_count = greedy_fill(current_adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
    
    # Local search: remove random edges, then re-greedy
    while time.time() < deadline:
        current_adj = best_adj[:]
        
        # Randomly remove 1 to 8 edges to escape local optima
        num_removals = random.randint(1, 8)
        edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges_list.append((u, v))
        
        if not edges_list:
            continue
            
        random.shuffle(edges_list)
        for u, v in edges_list[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
