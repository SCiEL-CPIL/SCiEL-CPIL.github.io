import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def is_valid(n, adj):
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Check if any w in common has edge to x in common
                    # Simpler: iterate w in common
                    w_list = [w for w in range(n) if (common >> w) & 1]
                    for i in range(len(w_list)):
                        w = w_list[i]
                        for j in range(i + 1, len(w_list)):
                            x = w_list[j]
                            if adj[w] & (1 << x):
                                return False, "K4"
        # Check Independent 6
        # This is O(n^6) if naive, but n is small.
        # Better: use bitmask for independent set check
        for combo in combinations(range(n), 6):
            # Check if all pairs are non-adjacent
            is_indep = True
            for i in range(6):
                for j in range(i + 1, 6):
                    if adj[combo[i]] & (1 << combo[j]):
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False, "I6"
        return True, "Valid"

    def sa_search(n, time_limit):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Precompute initial violations
        # We need a fast way to count violations or at least check validity
        # For SA, we need delta.
        
        def count_k4(adj):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        common = adj[u] & adj[v]
                        # Count edges in common
                        # For each w in common, count edges to other nodes in common
                        # This is O(n^3) per count. Too slow for every step.
                        # We must use incremental delta.
                        pass
            return 0 # Placeholder

        # We will implement the incremental delta logic inside the loop
        # For simplicity and robustness in this short snippet, 
        # we will use a "greedy repair" approach which is easier to verify correctness.
        # Greedy: While invalid, find a violation, remove an edge from it.
        
        # However, greedy repair might get stuck in a local minimum of "many violations".
        # SA is better. Let's implement a simplified SA with approximate deltas or just validity checks?
        # Validity check is O(n^3 + n^6). n=35. 35^6 is ~1.8e9. Too slow.
        
        # We MUST use the fast delta method.
        
        def calc_delta_k4(adj, u, v):
            # Returns change in K4 count if edge (u,v) is flipped
            # If adding: + (edges in N(u) cap N(v))
            # If removing: - (edges in N(u) cap N(v))
            common = adj[u] & adj[v]
            # Count edges in induced subgraph on common
            edge_count = 0
            # Iterate w in common
            c_list = [w for w in range(n) if (common >> w) & 1]
            for i in range(len(c_list)):
                w = c_list[i]
                # Count neighbors of w in common
                # adj[w] & common
                # Bit count
                edge_count += bin(adj[w] & common).count('1')
            return edge_count // 2

        def calc_delta_i6(adj, u, v):
            # If adding edge (u,v): destroy I6 containing u,v
            # If removing edge (u,v): create I6 containing u,v
            # Number of I6 containing u,v is number of independent 4-sets in 
            # D = NonNeighbors(u) cap NonNeighbors(v)
            
            # NonNeighbors mask
            all_mask = (1 << n) - 1
            non_u = all_mask ^ adj[u] ^ (1 << u)
            non_v = all_mask ^ adj[v] ^ (1 << v)
            D = non_u & non_v
            
            # Count independent 4-sets in G[D]
            d_list = [w for w in range(n) if (D >> w) & 1]
            count = 0
            # If len(d_list) < 4, return 0
            if len(d_list) < 4:
                return 0
            
            # Iterate 4-combinations of d_list
            # This is expensive. 
            # Optimization: Only do this if we are removing an edge (creating I6 is bad) 
            # or if we are adding an edge and we want to know how many we destroy.
            
            # For speed, we will approximate or limit the search.
            # If d_list is large, we sample? No, we need exact delta for SA.
            # Let's assume d_list is small enough (approx 10-15).
            # 15 choose 4 is 1365.
            for combo in combinations(d_list, 4):
                # Check if independent
                is_indep = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        if adj[combo[i]] & (1 << combo[j]):
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    count += 1
            return count

        # Initialize counts
        # We need initial K4 and I6 counts.
        # This is slow. Let's just run SA and track "energy" as sum of deltas?
        # No, we need absolute values to know when we hit 0.
        
        # Let's calculate initial counts once.
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    k4_count += calc_delta_k4(adj, u, v)
        k4_count //= 2 # Each K4 counted 6 times? No.
        # calc_delta_k4(u,v) returns number of K4s containing edge (u,v).
        # Each K4 has 6 edges. So sum over all edges / 6 = total K4s.
        # My loop sums over all edges.
        # So total_k4 = sum(delta) / 6.
        
        i6_count = 0
        # This is much harder to initialize.
        # Let's just use a validity check for the "accept" condition?
        # No, SA needs energy.
        
        # Alternative: Use a penalty function that is cheaper to compute.
        # E.g., just check if *any* K4 or I6 exists?
        # If we use a boolean penalty, we can't do SA properly.
        
        # Let's try a different strategy: 
        # 1. Generate random graph.
        # 2. While invalid:
        #    Find a specific K4 or I6.
        #    Flip an edge in it.
        #    This is "Greedy Repair".
        
        def find_violation(adj):
            # Find K4
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        common = adj[u] & adj[v]
                        c_list = [w for w in range(n) if (common >> w) & 1]
                        for i in range(len(c_list)):
                            w = c_list[i]
                            for j in range(i + 1, len(c_list)):
                                x = c_list[j]
                                if adj[w] & (1 << x):
                                    return (u, v, w, x), "K4"
            # Find I6
            for combo in combinations(range(n), 6):
                is_indep = True
                for i in range(6):
                    for j in range(i + 1, 6):
                        if adj[combo[i]] & (1 << combo[j]):
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return combo, "I6"
            return None, None

        best_adj = None
        best_score = float('inf')
        
        # We will run multiple restarts
        restarts = 0
        while time.time() < start_time + time_limit:
            restarts += 1
            # Re-initialize or continue from previous?
            # Let's restart with random for diversity
            if restarts > 1:
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Greedy Repair Loop
            steps = 0
            max_steps = 5000 # Limit per restart
            while steps < max_steps and time.time() < start_time + time_limit:
                steps += 1
                viol, vtype = find_violation(adj)
                if viol is None:
                    # Valid!
                    return adj
                # Pick an edge from the violation to flip
                if vtype == "K4":
                    u, v, w, x = viol
                    # Flip one of the edges
                    # e.g., (u,v)
                    a, b = u, v
                else:
                    # I6
                    # Pick two non-adjacent vertices to add an edge between
                    # e.g., viol[0], viol[1]
                    a, b = viol[0], viol[1]
                
                # Flip edge (a,b)
                adj[a] ^= (1 << b)
                adj[b] ^= (1 << a)
                
                # Check if this flip reduced the "badness"
                # We don't have a score, so we just hope.
                # To improve, we can check if the new graph has fewer violations?
                # That requires counting, which is slow.
                # Let's just do random flips if stuck?
                
                # Simple heuristic: If we flip and it's still invalid, continue.
                # If we loop, we might oscillate.
                # Add a random element: 10% chance to flip a random edge.
                if random.random() < 0.1:
                    ra = random.randint(0, n - 1)
                    rb = random.randint(0, n - 1)
                    if ra != rb:
                        adj[ra] ^= (1 << rb)
                        adj[rb] ^= (1 << ra)

        return None

    # Try n=35
    for n in [35, 34, 33, 32, 31, 30, 25, 20, 17]:
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        # Run SA/Greedy
        result_adj = sa_search(n, time_left - 2)
        
        if result_adj is not None:
            # Verify one last time
            valid, _ = is_valid(n, result_adj)
            if valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if result_adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
