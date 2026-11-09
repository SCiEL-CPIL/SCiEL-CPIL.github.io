import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency list as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges

    def find_k4s(adj, n):
        k4s = []
        # Iterate over all combinations of 4
        # Optimization: Only check if they form a K4
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)) and \
               (adj[b] & (1 << c)) and (adj[b] & (1 << d)) and \
               (adj[c] & (1 << d)):
                k4s.append(combo)
        return k4s

    def find_ind4s(adj, n):
        ind4s = []
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if not (adj[a] & (1 << b)) and not (adj[a] & (1 << c)) and not (adj[a] & (1 << d)) and \
               not (adj[b] & (1 << c)) and not (adj[b] & (1 << d)) and \
               not (adj[c] & (1 << d)):
                ind4s.append(combo)
        return ind4s

    def find_valid_neighbor_set(adj, n, k4s, ind4s, deadline):
        """
        Finds a set S subset of {0..n-1} such that:
        1. No K4 is a subset of S.
        2. No Ind4 is a subset of {0..n-1} \ S.
        Uses Min-Conflicts heuristic.
        """
        if time.time() > deadline - 5:
            return None
            
        # Precompute masks for fast checking
        k4_masks = [sum([1 << v for v in C]) for C in k4s]
        ind4_masks = [sum([1 << v for v in I]) for I in ind4s]
        
        # Count occurrences of each vertex in K4s and Ind4s
        k4_count = [0] * n
        ind4_count = [0] * n
        for m in k4_masks:
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                k4_count[v] += 1
                temp ^= lsb
        for m in ind4_masks:
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                ind4_count[v] += 1
                temp ^= lsb
        
        # Heuristic initialization: 
        # We want to avoid putting vertices with high k4_count in S (to avoid K4 in S)
        # We want to avoid putting vertices with high ind4_count OUT of S (to avoid Ind4 in comp(S))
        # So, if ind4_count[v] > k4_count[v], prefer v in S.
        
        best_S = None
        best_violations = float('inf')
        
        # Try multiple restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            S = 0
            if restart == 0:
                # Greedy
                for v in range(n):
                    if ind4_count[v] > k4_count[v]:
                        S |= (1 << v)
            else:
                # Random with bias
                for v in range(n):
                    # Prob of being in S
                    # If k4_count is high, we don't want it in S (prob low)
                    # If ind4_count is high, we want it in S (prob high)
                    bias = (ind4_count[v] - k4_count[v]) / 10.0
                    prob = 0.5 + bias
                    if random.random() < prob:
                        S |= (1 << v)
            
            # Calculate initial violations
            viol = 0
            for m in k4_masks:
                if (S & m) == m:
                    viol += 1
            for m in ind4_masks:
                if (S & m) == 0:
                    viol += 1
            
            if viol == 0:
                return S
                
            best_S = S
            best_violations = viol
            
            # Min-Conflicts Local Search
            # We keep track of violations per vertex to speed up updates
            # viol_v[v] = number of K4s fully in S containing v + number of Ind4s fully in comp(S) containing v
            # This is complex to update incrementally without iterating all constraints.
            # Given n <= 46 and |K4|, |Ind4| are manageable, we can just recalculate total viol per step if steps are few.
            # Or, we can use a simpler heuristic: Flip the vertex that seems most "problematic".
            
            # Let's use a simple SA on the S bitmask
            T = 5.0
            cooling = 0.99
            curr_viol = viol
            curr_S = S
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                if curr_viol == 0:
                    return curr_S
                    
                v = random.randint(0, n - 1)
                new_S = curr_S ^ (1 << v)
                
                # Calculate delta in violations
                # This requires checking all K4s and Ind4s containing v.
                # Precomputing lists of K4s/Ind4s per vertex would help.
                # For now, simple check is O(|K4| + |Ind4|).
                # To speed up, we can just recalculate total viol if we do it rarely, 
                # but here we do it every step. 
                # Let's optimize by precomputing lists.
                
                # Since we are in a tight loop, let's just do a fast approximate check or 
                # accept the cost if |K4|+|Ind4| is small enough.
                # For n=40, |K4| can be in the thousands. 2000 steps * 5000 checks = 10M ops. OK.
                
                new_viol = 0
                for m in k4_masks:
                    if (new_S & m) == m:
                        new_viol += 1
                for m in ind4_masks:
                    if (new_S & m) == 0:
                        new_viol += 1
                        
                delta = new_viol - curr_viol
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    curr_S = new_S
                    curr_viol = new_viol
                    if curr_viol < best_violations:
                        best_violations = curr_viol
                        best_S = curr_S
                T *= cooling
                
            if best_violations == 0:
                return best_S
                
        return best_S if best_violations == 0 else None

    # Iteratively add vertices
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 10:
            break
        
        k4s = find_k4s(adj, n)
        ind4s = find_ind4s(adj, n)
        
        added = False
        attempts = 0
        while attempts < 10 and not added and time.time() < deadline - 10:
            attempts += 1
            S = find_valid_neighbor_set(adj, n, k4s, ind4s, deadline)
            if S is not None:
                # Add new vertex n
                new_adj = adj + [0]
                for v in range(n):
                    if S & (1 << v):
                        new_adj[n] |= (1 << v)
                        new_adj[v] |= (1 << n)
                adj = new_adj
                n += 1
                added = True
                # Update k4s and ind4s for next iteration? 
                # We recalculate at the start of the outer loop.
                
        if not added:
            # Try edge flips to break deadlock
            if time.time() > deadline - 5:
                break
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j or i > j:
                    continue
                # Flip edge (i, j)
                old_adj = adj[:]
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
                k4s_new = find_k4s(adj, n)
                ind4s_new = find_ind4s(adj, n)
                S = find_valid_neighbor_set(adj, n, k4s_new, ind4s_new, deadline)
                if S is not None:
                    new_adj = adj + [0]
                    for v in range(n):
                        if S & (1 << v):
                            new_adj[n] |= (1 << v)
                            new_adj[v] |= (1 << n)
                    adj = new_adj
                    n += 1
                    added = True
                    break
                else:
                    adj = old_adj
                    
            if not added:
                break
                
    edges = get_edges(adj, n)
    return n, edges

# EVOLVE-BLOCK-END
