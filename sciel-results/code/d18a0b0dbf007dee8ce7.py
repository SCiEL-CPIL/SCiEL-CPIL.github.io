"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    N = 256

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Factorize each number 1..N into its set of distinct prime factors
    def factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Build the set of all candidate numbers (1..256)
    candidates = list(range(1, N + 1))
    cand_factors = {}
    for c in candidates:
        if c == 1:
            cand_factors[c] = set()
        else:
            cand_factors[c] = factorize(c)

    # Build conflict graph: edge between u and v if gcd(u,v) > 1
    # We want Maximum Weight Independent Set, which is equivalent to Maximum Weight Clique in the complement graph.
    # Complement graph: edge between u and v if gcd(u,v) == 1.
    
    n_nodes = N  # Nodes are 1..256, index 0..255 corresponds to value 1..256
    # weights[i] = value of node i+1
    weights = [i + 1 for i in range(n_nodes)]
    
    # Adjacency matrix for the "coprime" graph (complement of conflict graph)
    # adj[u][v] = True if gcd(u+1, v+1) == 1
    # Using a list of sets or bitmasks for efficiency
    adj = [set() for _ in range(n_nodes)]
    
    for i in range(n_nodes):
        vi = i + 1
        for j in range(i + 1, n_nodes):
            vj = j + 1
            if math.gcd(vi, vj) == 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Maximum Weight Clique on this graph.
    # We use Bron-Kerbosch with pivot, modified for weighted clique (branch and bound).
    
    best_clique = []
    best_weight = 0
    
    # To speed up, we can order nodes by weight descending?
    # Actually, standard Bron-Kerbosch works on the graph.
    # For weighted clique, we use a branch and bound where we prune if current_weight + sum(remaining_weights) <= best_weight.
    
    # Let's implement a recursive search.
    # State: 
    #   R: set of nodes in current clique
    #   P: set of candidates that can be added to R (must be adjacent to all in R)
    #   X: set of nodes already processed (not strictly needed for standard BK, but for weighted we might need care)
    
    # Optimization: Precompute sum of weights for any subset? No, too expensive.
    # Instead, at each step, calculate the upper bound as current_weight + sum(weights[v] for v in P).
    
    # To make it faster, we can sort P by weight descending and process.
    
    # Let's use a bitmask representation for P and X if possible? 256 bits is too large for standard int in some contexts? 
    # Python ints are arbitrary precision, so 256 bits is fine.
    
    # Let's use bitmasks for adjacency.
    # adj_mask[i] is a bitmask where bit j is 1 if i and j are adjacent (coprime).
    adj_mask = [0] * n_nodes
    for i in range(n_nodes):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
        
    # Function to get sum of weights of a set represented by bitmask
    # This is expensive if done naively. 
    # However, n=256 is small. 
    # Let's try a simpler recursive approach with sets first, and optimize if too slow.
    # Given 90s budget, sets might be okay if pruning is good.
    
    # But bitmasks are much faster for intersection.
    
    # Let's stick to bitmasks.
    # We need to iterate over bits in a mask.
    
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
            
    def popcount(mask):
        return bin(mask).count('1')
        
    # Precompute weight sum for small masks? No.
    # We will maintain current_weight.
    # Upper bound: current_weight + sum(weights[i] for i in P).
    # To speed up sum(weights[i] for i in P), we can precompute nothing, just iterate.
    # Or, we can use a heuristic: sort P by weight descending and stop adding to bound if we exceed best? No, that's not an upper bound.
    
    # Let's implement the search.
    
    # Initial P: all nodes.
    # We can remove nodes that are not connected to anything? No, 1 is connected to all.
    
    # Order nodes by weight descending to find good solutions early?
    # The order of expansion in BK affects performance.
    # Let's try expanding nodes in descending order of weight.
    
    # We will use a recursive function:
    # expand(R_mask, P_mask, X_mask, current_weight)
    
    # To avoid deep recursion issues (max depth 256), Python recursion limit might be hit.
    # sys.setrecursionlimit(10000)
    
    sys.setrecursionlimit(10000)
    
    # Global variables for best
    global_best_weight = 0
    global_best_mask = 0
    
    def search(R_mask, P_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if time.time() > deadline:
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound check
        # Sum of weights in P
        bound = current_weight
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            pm ^= lsb
            
        if bound <= global_best_weight:
            return
            
        # Pivot selection
        # Choose pivot u from P union X (X is empty in this simplified BK variant, or we can use P only for pivot)
        # Standard BK uses P U X. Since we don't track X explicitly in this state (we just pass P of candidates),
        # we can just pick a pivot from P to reduce branching.
        # Actually, the standard optimization is to choose a pivot u in P U X, and only branch on vertices in (P \ N(u)).
        # Since X is not tracked, we can approximate by picking u in P.
        
        # Let's pick the vertex in P with the highest degree in P (or highest weight) as pivot.
        # For weighted clique, picking high weight pivot might be good.
        
        pivot_idx = -1
        pivot_degree = -1
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            # Degree of idx in P
            deg = popcount(P_mask & adj_mask[idx])
            if deg > pivot_degree:
                pivot_degree = deg
                pivot_idx = idx
            pm ^= lsb
            
        # If no pivot (P was empty, handled above), this shouldn't happen.
        
        # Branch on vertices in P that are NOT adjacent to pivot
        # Wait, the theorem says: there is a maximal clique containing R and a subset of P\N(u).
        # So we iterate over v in P such that v is NOT adjacent to u.
        
        # If pivot covers all P, we are done? No, we still need to check if current R is a clique.
        # Actually, if P \ N(u) is empty, it means all vertices in P are adjacent to u.
        # Then any maximal clique extending R must contain u? No.
        # The property is: for any maximal clique Q containing R, Q \ R is a subset of P \ N(u) for some u in P U X?
        # No, the standard BK pivot optimization:
        # Pick u in P U X.
        # For each v in P \ N(u):
        #     expand(R U {v}, P \ (N(v) U {v}), ...)
        #     P = P \ {v}
        #     X = X U {v}
        
        # If we don't track X, we can't do the standard BK correctly to avoid duplicates?
        # Actually, for finding *a* maximum weight clique, we don't need to enumerate all maximal cliques, we just need to search the space.
        # The pivot optimization reduces the branching factor.
        
        # Let's implement the standard BK with X.
        
        # Revised search signature:
        # search(R_mask, P_mask, X_mask, current_weight)
        
        # We'll restart the search logic below with X.
        
    # Let's rewrite the search function properly with X.
    
    def bk_search(R_mask, P_mask, X_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if time.time() > deadline:
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound
        bound = current_weight
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            pm ^= lsb
            
        if bound <= global_best_weight:
            return
            
        # Pivot selection from P U X
        pivot_candidates = P_mask | X_mask
        pivot_idx = -1
        max_deg = -1
        
        pm = pivot_candidates
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            # Degree in P (neighbors of idx that are in P)
            deg = popcount(P_mask & adj_mask[idx])
            if deg > max_deg:
                max_deg = deg
                pivot_idx = idx
            pm ^= lsb
            
        # If no pivot found (shouldn't happen if P is not empty), pick any from P
        if pivot_idx == -1:
            lsb = P_mask & -P_mask
            pivot_idx = lsb.bit_length() - 1
            
        # Branch on vertices in P that are NOT adjacent to pivot
        # These are the candidates that could form a clique without the pivot.
        # Vertices in P that ARE adjacent to pivot are handled in the recursive calls?
        # No, the logic is:
        # Every maximal clique containing R is either:
        # 1. Contains a vertex from P \ N(pivot)
        # 2. Is a subset of R U N(pivot) (but we are expanding, so we look for vertices to add)
        
        # Standard BK:
        # for v in P \ N(pivot):
        #     expand(R U {v}, P \ (N(v) U {v}), X \ N(v))
        #     P = P \ {v}
        #     X = X U {v}
        
        # Note: We must iterate over P \ N(pivot).
        # Let's get the set of vertices to iterate.
        # verts_to_iter = P_mask & ~adj_mask[pivot_idx]
        # But we also need to remove them from P and add to X as we go.
        
        verts_to_iter = P_mask & ~adj_mask[pivot_idx]
        
        # To improve pruning, we might want to iterate in order of decreasing weight.
        # Let's collect them and sort.
        verts_list = []
        vm = verts_to_iter
        while vm:
            lsb = vm & -vm
            idx = lsb.bit_length() - 1
            verts_list.append(idx)
            vm ^= lsb
            
        verts_list.sort(key=lambda x: weights[x], reverse=True)
        
        for v in verts_list:
            # Check if v is still in P (it might have been removed? No, we are iterating a snapshot)
            # But we need to update P and X for subsequent iterations.
            # The loop structure in BK usually modifies P and X.
            
            # Current P and X are passed by value (bitmasks are immutable in Python int context if we create new ones)
            
            # New R
            new_R = R_mask | (1 << v)
            
            # New P: neighbors of v in current P, excluding v
            new_P = P_mask & adj_mask[v] & ~(1 << v)
            
            # New X: neighbors of v in current X, excluding v
            new_X = X_mask & adj_mask[v] & ~(1 << v)
            
            # Update weight
            new_weight = current_weight + weights[v]
            
            # Recursive call
            bk_search(new_R, new_P, new_X, new_weight)
            
            # Update P and X for the next iteration
            # P = P \ {v}
            P_mask &= ~(1 << v)
            # X = X U {v}
            X_mask |= (1 << v)
            
            if time.time() > deadline:
                return
                
    # Initial call
    # R = 0
    # P = all nodes (1..256)
    # X = 0
    initial_P = (1 << n_nodes) - 1
    
    # Optimization: We can start with a greedy heuristic to set a good initial lower bound.
    # Greedy: pick node with max weight, add to clique, remove non-neighbors.
    # Repeat.
    
    def greedy_clique():
        P = initial_P
        R = 0
        w = 0
        while P:
            # Pick max weight node in P
            best_node = -1
            best_w = -1
            pm = P
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_node = idx
                pm ^= lsb
                
            if best_node == -1:
                break
                
            R |= (1 << best_node)
            w += weights[best_node]
            
            # Update P: keep only neighbors of best_node
            P = P & adj_mask[best_node] & ~(1 << best_node)
            
        return R, w
        
    init_R, init_w = greedy_clique()
    global_best_weight = init_w
    global_best_mask = init_R
    
    # Run exact search
    bk_search(0, initial_P, 0, 0)
    
    if time.time() > deadline:
         # Return best found so far
         pass
    
    # Extract result from global_best_mask
    result = []
    mask = global_best_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
        
    # Ensure distinct and in range
    result = list(set(result))
    
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback if something went wrong
                return [1] + [p for p in primes]
                
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
