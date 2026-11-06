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
        3. For any neighbor w of u (w != v), |N(v) \cap N(w)| <= 1.
        4. For any neighbor w of v (w != u), |N(u) \cap N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # Fast pre-check: if u has any neighbor w such that v and w already share 2+ common neighbors,
        # adding (u,v) would make u the 3rd common neighbor. Similarly for v.
        # We can check this more efficiently: for each w in N(u)\{v}, check popcount(adj[v] & adj[w]) > 1.
        # Pre-check: if |N(u) \ N(v)| is large and adj[v] is dense, likely to fail.
        # Actually, let's just do the check but with an early-exit optimization.
        
        # 3. Check constraints on neighbors of u
        w_mask = adj[u] & ~(1 << v)  # exclude v from consideration
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        w_mask = adj[v] & ~(1 << u)  # exclude u from consideration
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
        Uses time-based cooling instead of step-based for consistent behavior.
        """
        if time.time() > deadline - 5:
            return count
        
        sa_start = time.time()
        sa_duration = 8.0  # seconds per SA phase
        sa_end = min(sa_start + sa_duration, deadline - 2)
        
        T = 1.5
        min_T = 0.01
        # Use time-based cooling: T decreases exponentially over sa_duration
        total_time = sa_duration
        cooling_rate = (min_T / T) ** (1.0 / total_time) if total_time > 0 else 1.0
        
        steps_since_improve = 0
        max_steps_no_improve = 200
        
        while time.time() < sa_end and steps_since_improve < max_steps_no_improve:
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
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            elif move_type < 0.8:
                # Remove a present edge and then refill (kick move)
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    old_count = count
                    remove_edge(adj, u, v)
                    count -= 1
                    
                    # Try to greedily add back edges
                    count = greedy_fill(adj, count)
                    if count > old_count:
                        steps_since_improve = 0
                    elif count == old_count:
                        # Neutral, accept with probability based on T
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            # Revert: we can't easily revert greedy_fill, so accept
                            steps_since_improve += 1
                    else:
                        # Got worse, but accept with probability T to escape
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
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
                        # Accept neutral moves to explore.
                        steps_since_improve = 0
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        count += 1
                        steps_since_improve += 1
            
            # Cool based on elapsed time
            elapsed = time.time() - sa_start
            T = max(min_T, T * cooling_rate ** (1.0 / 60.0))  # approximate per-step cooling
            
        # Final greedy fill
        count = greedy_fill(adj, count)
        
        return count

    def smart_initialization():
        """
        Initialize with a dense starting graph: start with all edges, then remove
        edges that violate K2,3-freeness. This gives a much denser starting point
        than random greedy fill.
        """
        adj = [0] * n
        count = 0
        # Start with a random subset to avoid symmetric traps
        # Use a random permutation and greedily add
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def smart_initialization_v2():
        """
        Alternative: start with complement of a random matching, then greedily add.
        This starts with ~190 edges minus a few, giving a very dense seed.
        """
        adj = [0] * n
        count = 0
        # Start with all edges
        for i in range(n):
            for j in range(i + 1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        count = n * (n - 1) // 2
        
        # Now remove edges that cause violations.
        # A violation is a pair (u,v) with 3+ common neighbors.
        # We need to remove enough edges to fix all violations.
        # Strategy: for each violating pair, remove one edge incident to a common neighbor.
        
        changed = True
        max_iter = 50
        it = 0
        while changed and it < max_iter and time.time() < deadline:
            changed = False
            it += 1
            # Find a violating pair
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() > 2:
                        # Remove a random edge to break the violation
                        # Remove edge (u, v) or an edge from a common neighbor
                        # Let's remove a random edge among the common neighbors' connections
                        c_list = []
                        cm = common
                        while cm:
                            lsb = cm & -cm
                            w = lsb.bit_length() - 1
                            c_list.append(w)
                            cm ^= lsb
                        # Remove a random edge incident to one of the common neighbors
                        w = random.choice(c_list)
                        # Remove edge (w, u) or (w, v)
                        if random.random() < 0.5:
                            remove_edge(adj, w, u)
                        else:
                            remove_edge(adj, w, v)
                        count -= 1
                        changed = True
                        break
                if changed:
                    break
        return adj, count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Alternate between initialization strategies
        if restarts % 3 == 0:
            # Random greedy fill
            random.shuffle(all_edges)
            adj = [0] * n
            count = 0
            for u, v in all_edges:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
        elif restarts % 3 == 1:
            # Dense start with violation removal
            adj, count = smart_initialization_v2()
        else:
            # Random greedy with different seed
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
