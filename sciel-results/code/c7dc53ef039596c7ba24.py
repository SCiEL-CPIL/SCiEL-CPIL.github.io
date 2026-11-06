import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper functions
    
    def count_k4(adj):
        """
        Count the number of K4s in the graph.
        adj: list of ints, where adj[i] is a bitmask of neighbors of i.
        """
        n = len(adj)
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Common neighbors
                    common = adj[u] & adj[v]
                    # Count edges in common
                    # Iterate over bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other node in common
                        # We can just check if adj[w] & common has any bits other than w itself?
                        # No, adj[w] contains neighbors.
                        # We need to know if there is an edge (w, x) where x is in common.
                        # Since we are iterating w, we can check adj[w] & (common & ~(1<<w))
                        # But common doesn't contain u or v.
                        # So just check if adj[w] & common is non-zero.
                        # Wait, adj[w] might contain w? No, simple graph.
                        # But adj[w] & common will count each edge twice if we're not careful?
                        # No, we just want to know if ANY edge exists.
                        if adj[w] & common:
                            count += 1
                            break # Just count 1 for this pair (u,v) if it forms at least one K4? 
                            # No, the task says "no K4". So if there is at least one, it's a violation.
                            # But for SA, counting the number of K4s is a better gradient.
                            # Let's count all K4s involving (u,v).
                            # Number of K4s containing (u,v) is the number of edges in the subgraph induced by common.
                            # Let's compute that.
                            edge_count = 0
                            c2 = common
                            while c2:
                                lsb2 = c2 & -c2
                                w2 = lsb2.bit_length() - 1
                                c2 ^= lsb2
                                # Check edges from w2 to other nodes in common
                                # To avoid double counting, only check w2 < x?
                                # Or just count all and divide by 2?
                                # adj[w2] & common will count edges to all neighbors in common.
                                # Since the graph is undirected, each edge (w2, x) is counted once for w2 and once for x.
                                # So sum of (adj[w2] & common).bit_count() for all w2 in common / 2.
                                edge_count += (adj[w2] & common).bit_count()
                            count += edge_count // 2
        return count

    def has_is6(adj):
        """
        Check if there is an independent set of size 6.
        Uses backtracking with pruning.
        """
        n = len(adj)
        # Greedy check first
        # Try to find a large IS
        def greedy_is(start=0):
            remaining = (1 << n) - 1
            count = 0
            while remaining:
                # Pick a vertex with minimum degree in remaining?
                # Or just pick the first one for speed
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                count += 1
                remaining &= ~(adj[v] | lsb)
                if count >= 6:
                    return True
            return False
        
        if greedy_is():
            return True
        
        # Backtracking
        # We need to check if alpha >= 6
        # We can use a recursive function that tries to build an IS of size 6
        # State: current set (bitmask), size, candidate set
        # To be fast, we only need to find ONE.
        
        # Order vertices by degree?
        # Let's just do a simple DFS
        
        # Precompute non-neighbors?
        # non_adj[i] = ~adj[i] & ((1<<n)-1) & ~(1<<i)
        
        def dfs(current_size, candidates):
            if current_size == 6:
                return True
            if not candidates:
                return False
            # Pruning: if max possible size < 6, return False
            # Max possible size is current_size + popcount(candidates)
            if current_size + candidates.bit_count() < 6:
                return False
            
            # Pick a vertex
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates_without_v = candidates ^ lsb
            
            # Branch 1: Include v
            # New candidates: candidates_without_v & non_adj[v]
            non_adj_v = (~adj[v]) & ((1 << n) - 1) & ~(1 << v)
            new_candidates = candidates_without_v & non_adj_v
            if dfs(current_size + 1, new_candidates):
                return True
            
            # Branch 2: Exclude v
            if dfs(current_size, candidates_without_v):
                return True
            
            return False
        
        return dfs(0, (1 << n) - 1)

    def get_best_graph(n):
        """
        Run SA to find a valid graph of size n.
        Returns (adj, valid)
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Calculate initial cost
        k4_count = count_k4(adj)
        is6_exists = has_is6(adj)
        
        # SA Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        max_iters = 5000 # Adjust based on time
        
        best_adj = adj[:]
        best_cost = (k4_count, 1 if is6_exists else 0)
        
        current_cost = best_cost
        current_adj = adj
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Calculate delta K4
            old_k4 = count_k4(current_adj) # This is expensive!
            # Optimization: Only update K4 count incrementally?
            # For now, let's just recalculate if it's fast enough.
            # n=35, edges ~ 300. count_k4 is O(E * deg^2). Might be slow.
            
            # Let's try to flip
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            new_k4 = count_k4(current_adj)
            new_is6 = has_is6(current_adj)
            
            new_cost = (new_k4, 1 if new_is6 else 0)
            
            # Acceptance
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = current_adj[:]
                    if best_cost == (0, 0):
                        break
            else:
                # Delta cost
                # We treat cost as a single number for SA
                # Cost = 100 * k4 + is6
                old_val = 100 * current_cost[0] + current_cost[1]
                new_val = 100 * new_cost[0] + new_cost[1]
                delta = new_val - old_val
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        return best_adj, best_cost == (0, 0)

    # Main Loop
    best_n = 17
    best_adj = None
    
    # Start with Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17
    
    # Try to increase n
    for n in range(18, 36):
        if time.time() > deadline - 10.0:
            break
        
        # Run SA for this n
        # Allocate more time for larger n?
        # Just run get_best_graph
        adj, valid = get_best_graph(n)
        
        if valid:
            best_n = n
            best_adj = adj
            # If we reached 35, we can stop
            if n == 35:
                break
        else:
            # If we fail, we might still try again with a different seed?
            # Or just move on?
            # Moving on might be risky if n=18 fails but n=19 works? Unlikely.
            # Usually if you can't find a valid graph for n, you can't for n+1.
            # But maybe the search space is just hard.
            # Let's try one more time if we have time.
            if time.time() < deadline - 20.0:
                adj2, valid2 = get_best_graph(n)
                if valid2:
                    best_n = n
                    best_adj = adj2
                    break
            else:
                break

    # Convert best_adj to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
