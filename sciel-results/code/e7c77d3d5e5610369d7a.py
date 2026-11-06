import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s to be safe.
    deadline = start_time + 145.0
    
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
        2. |N(u) ∩ N(v)| <= 2.
        3. For all w in N(u), |N(v) ∩ N(w)| <= 1.
        4. For all w in N(v), |N(u) ∩ N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
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
        Uses a shuffled list to avoid bias.
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
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        """
        if time.time() > deadline - 5:
            return count
        
        T = 5.0  # Start high to allow exploration
        cooling_rate = 0.999
        min_T = 0.1
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        # Accept improvement immediately
                    else:
                        # Invalid addition, skip
                        pass
            elif move_type < 0.85:
                # Try to remove a present edge and then refill (Kick)
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    
                    # Try to greedily add back edges
                    # This is a "kick" move that can jump out of local minima
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
                        # Accept neutral moves to explore (Metropolis criterion for delta=0 is always accept)
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
            
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
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
        
        # Simulated Annealing
        count = simulated_annealing(adj, count)
        
        # Final greedy fill
        count = greedy_fill(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # If we are finding very good solutions, we can try to run SA longer on the best one
        if best_count >= 125 and time.time() > deadline - 30:
             # Polish the best solution found so far
             adj = best_adj[:]
             count = best_count
             count = simulated_annealing(adj, count)
             count = greedy_fill(adj, count)
             if count > best_count:
                 best_count = count
                 best_adj = adj[:]
             break

    # Construct the final list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick self-test to ensure it runs and returns valid format
    edges = construct()
    # Verify K2,3 freeness for debugging (removed in final output for speed)
    # n = 20
    # adj = [0]*n
    # for u,v in edges:
    #     adj[u] |= 1<<v
    #     adj[v] |= 1<<u
    # for u in range(n):
    #     for v in range(u+1, n):
    #         if (adj[u] & adj[v]).bit_count() >= 3:
    #             print(f"Invalid: {u}, {v} have >=3 common neighbors")
    #             sys.exit(1)
    # print(f"Edges: {len(edges)}")
