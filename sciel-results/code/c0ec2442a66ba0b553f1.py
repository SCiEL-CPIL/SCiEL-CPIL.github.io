import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        
        Constraints for adding (u, v):
        1. u and v must not already be connected.
        2. The number of common neighbors of u and v must be <= 2.
           (If it were 3, adding (u,v) would create a K2,3 with the 3 common neighbors as the other partition).
           Actually, K2,3 has partitions of size 2 and 3.
           If we add (u,v), u and v are in the partition of size 2.
           Their common neighbors are the candidates for the partition of size 3.
           So we need |N(u) \cap N(v)| <= 2.
           
        3. For any neighbor w of u (w != v), adding (u,v) makes u a common neighbor of v and w.
           So we must ensure that v and w do not already have 2 common neighbors.
           If they did, adding (u,v) would make u the 3rd common neighbor, creating K2,3.
           So for all w in N(u), |N(v) \cap N(w)| <= 1.
           
        4. Similarly, for any neighbor w of v (w != u), adding (u,v) makes v a common neighbor of u and w.
           So we must ensure that u and w do not already have 2 common neighbors.
           So for all w in N(v), |N(u) \cap N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        # N(u) and N(v) do not include u or v themselves in the bitmask representation 
        # because we only set bits for other vertices.
        # Wait, adj[u] contains v if edge exists. It does not contain u.
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 3. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # We need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # We need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        """
        Try to add as many edges as possible from the missing list.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Don't break, keep trying to add more in this pass
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        """
        if time.time() > deadline - 5:
            return count
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        # Pre-compute masks for speed? No, adj changes.
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        # Accept improvement
                        # T *= cooling_rate # Don't cool on success? Or cool slowly.
                    else:
                        # Invalid addition, skip
                        pass
            elif move_type < 0.8:
                # Try to remove a present edge and then refill
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    
                    # Try to greedily add back edges
                    # We allow adding multiple edges to potentially recover or exceed count
                    # This is a "kick" move
                    count = greedy_fill(adj, count)
            else:
                # Swap: Remove one, add one specific missing
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    
                    # Temporarily remove
                    remove_edge(adj, u1, v1)
                    count -= 1
                    
                    # Try to add
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                        # Count stays same, but structure changed.
                        # Accept with some probability? 
                        # Since we want to maximize edges, keeping same count is neutral.
                        # But it might open up future improvements.
                        # Let's always accept neutral moves to explore.
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
            
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Initialize with random greedy fill
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        
        # Local search to fill in missing edges
        count = greedy_fill(adj, count)
        
        # Simulated annealing
        count = simulated_annealing(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 15:
            break
        
        # Optional: If we found a very good solution, we can reduce restarts or just continue
        # The time limit handles this.

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    # Verify validity
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                valid = False
                print(f"Invalid: {u}, {v} have {common.bit_count()} common neighbors")
                break
        if not valid:
            break
    
    print(f"Valid: {valid}")
