import time
from math import gcd, lcm
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # 1. Precompute Conflict Graph
    # conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # We want an independent set in this graph (no edges between selected nodes)
    conflict = [[False] * n for _ in range(n)]
    # Also store neighbors for faster traversal
    neighbors = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(values[i], values[j]) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Weights are 1/value
    weights = [1.0 / values[i] for i in range(n)]
    
    # 2. Branch and Bound Solver
    # State: 
    #   P: set of available vertices (not yet decided)
    #   X: set of excluded vertices (decided not to take, or neighbors of taken)
    #   C: set of chosen vertices
    # Upper Bound:
    #   Sum of weights in C + Fractional Knapsack Bound on P
    #   To compute FK bound: sort P by weight/size? No, standard FK:
    #   Sort items in P by weight descending. 
    #   Wait, the constraint is "independent set". 
    #   A common relaxation for MWIS is to ignore the independence constraint 
    #   and just take all remaining nodes? That's too loose.
    #   Better relaxation: The maximum weight independent set in a graph 
    #   is bounded by the maximum weight clique in the complement? No.
    #   
    #   Let's use a simpler but effective bound for sparse graphs:
    #   The sum of weights of all remaining nodes in P is an upper bound.
    #   This is valid but might be weak.
    #   
    #   Improved Bound:
    #   For each node v in P, we can either take it (and remove its neighbors from P)
    #   or not take it.
    #   A standard bound for MWIS is based on the "clique cover" or "independent set" properties.
    #   
    #   Let's try a specific heuristic bound:
    #   Sort P by weight descending.
    #   Greedily pick a maximal independent set from P to get a lower bound? 
    #   No, we need an UPPER bound to prune.
    #   
    #   Upper Bound Strategy:
    #   1. Sum of weights of all nodes in P. (Weak)
    #   2. For each node v in P, the max weight independent set is at most 
    #      max( w[v] + MWIS(P \ N[v]), MWIS(P \ {v}) ).
    #      This is the recurrence. 
    #      
    #   To make it fast, we need a good upper bound for the subproblem.
    #   One common bound: The sum of weights of a maximal independent set 
    #   in the subgraph induced by P is a LOWER bound. 
    #   An UPPER bound can be obtained by solving the LP relaxation? 
    #   Or simply: The sum of weights of all nodes in P is an upper bound.
    #   
    #   Let's refine the "Sum of weights" bound.
    #   If we have a set of conflicting nodes, we can't take all.
    #   Consider the clique number? 
    #   
    #   Alternative: Use the "Weighted Coloring" bound? 
    #   Too complex to implement quickly.
    #   
    #   Let's stick to a recursive B&B with a simple but valid bound:
    #   UB = sum(weights[i] for i in P)
    #   This is valid. Is it tight enough?
    #   For small N=255, if the graph is sparse, P remains large, sum is large.
    #   However, as we pick high weight nodes, we remove their neighbors.
    #   
    #   Let's add a stronger bound:
    #   The maximum weight of an independent set in G[P] is <= 
    #   The maximum weight of a set of nodes such that no two are adjacent.
    #   
    #   Heuristic Bound:
    #   Sort P by weight descending: v1, v2, ..., vk.
    #   If v1 and v2 are not adjacent, we can take both.
    #   If they are adjacent, we can take at most one.
    #   This suggests a "packing" bound.
    #   
    #   Actually, a very effective bound for MWIS is:
    #   UB = sum_{v in P} w[v] / (1 + degree_P(v)/2) ? No.
    #   
    #   Let's use the "Clique Cover" duality? 
    #   In a perfect graph, MWIS = MWClique in complement. 
    #   Our graph is not necessarily perfect.
    #   
    #   Let's try a different approach: **Maximum Weight Independent Set via Integer Programming?**
    #   No, no external solvers.
    #   
    #   Let's optimize the B&B with a better ordering and bound.
    #   Bound: 
    #   Let P be the set of available vertices.
    #   Calculate the sum of weights of all vertices in P.
    #   Also, calculate the sum of weights of a greedy independent set in P (Lower Bound).
    #   If LB_global + UB_sub > Current_Best, prune.
    #   
    #   Wait, the standard B&B for MWIS:
    #   f(P, C)
    #   if P is empty: return sum(weights in C)
    #   UB = sum(weights in C) + sum(weights in P)
    #   if UB <= Best: return
    #   
    #   Pick v in P with highest weight.
    #   Option 1: Include v. New P = P \ (N[v] U {v}). New C = C U {v}.
    #   Option 2: Exclude v. New P = P \ {v}. New C = C.
    #   
    #   This is exponential. With N=255, it will fail unless pruning is extremely effective.
    #   The "Sum of weights" bound is very loose.
    #   
    #   Let's use a **Lagrangian Relaxation** bound or a **Greedy Bound**.
    #   A common tight bound for MWIS is based on the "independent set" relaxation:
    #   Sort vertices in P by weight descending.
    #   Let G' be the subgraph induced by P.
    #   The weight of the maximum independent set is at most the weight of the 
    #   maximum weight set of vertices such that no two are adjacent? That's the problem.
    #   
    #   Let's try a different angle: **The graph structure**.
    #   Conflicts exist if lcm(a,b) <= 256.
    #   If a > 128, lcm(a,b) >= a > 128. 
    #   If a > 256/2 = 128, then for any b, lcm(a,b) is a multiple of a.
    #   If a is prime > 128, lcm(a,b) = a*b (if b!=a) or a.
    #   Wait, if a > 128, lcm(a,b) > 128. 
    #   Is it always > 256?
    #   If a=129, b=1. lcm=129. But b>=2.
    #   If a=129, b=2, lcm=258 > 256.
    #   If a=129, b=3, lcm=387 > 256.
    #   If a=130, b=2, lcm=130 <= 256. Conflict!
    #   So large numbers can still conflict with small divisors.
    #   
    #   However, numbers > 128 have no multiples in [2, 256] except themselves.
    #   So if a > 128, its only conflict with b is if b divides a or a divides b (impossible) or they share a factor such that lcm <= 256.
    #   Since a > 128, 2a > 256. So lcm(a,b) cannot be a multiple of a unless it is a itself.
    #   So lcm(a,b) = a * b / gcd(a,b).
    #   If gcd(a,b) = g, then lcm = a/g * b.
    #   For lcm <= 256, we need a/g * b <= 256.
    #   Since a > 128, a/g must be small.
    #   If g = a, then b must be a (not distinct).
    #   If g = a/2, then a/2 * b <= 256 => b <= 512/a < 4.
    #   So conflicts for large a are limited to small b.
    #   
    #   This sparsity suggests that **Greedy** might actually be very good, 
    #   but the prompt asks for a DIFFERENT strategy.
    #   
    #   Let's implement a **Memetic Algorithm** with **Simulated Annealing** and **2-Swap** moves, 
    #   but with a **Constructive Lower Bound** to guide the search.
    #   Actually, the prompt specifically suggested "Branch-and-Bound ... fractional knapsack".
    #   Let's try to make B&B work by using a **Stronger Upper Bound**.
    #   
    #   Upper Bound Idea:
    #   The weight of the maximum independent set in G[P] is at most 
    #   the sum of weights of the vertices in P divided by the clique cover number?
    #   No.
    #   
    #   Let's use the **Greedy Independent Set** as a Lower Bound (LB).
    #   And use **Sum of Weights** as Upper Bound (UB).
    #   To make UB tighter:
    #   For each vertex v in P, if we include v, we lose neighbors.
    #   This is the recursion.
    #   
    #   Given the time limit, a pure B&B might TLE.
    #   Let's hybridize:
    #   1. Run a fast Greedy to get a good initial LB.
    #   2. Run Local Search to improve LB.
    #   3. Run B&B with the LB as the cutoff.
    #   If B&B finds a better solution, update LB.
    #   If B&B TLEs, return best LB.
    #   
    #   To speed up B&B:
    #   - Use bitsets for P and neighbors.
    #   - Order vertices by weight descending.
    #   
    #   Implementation of B&B with Bitsets:
    
    # Precompute bitmasks for neighbors
    # n=255 fits in 4 64-bit ints or just Python ints (arbitrary precision)
    neighbor_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        neighbor_masks[i] = mask
        
    # Sort vertices by weight descending for branching order
    # We will branch on the highest weight available vertex.
    # To do this efficiently, we maintain a list of available vertices.
    
    best_score = 0.0
    best_set = []
    
    # Initial Greedy Solution
    def greedy_solve(avail_mask):
        S = []
        score = 0.0
        mask = avail_mask
        while mask:
            # Find highest weight vertex in mask
            # Iterate through sorted indices
            v = -1
            max_w = -1.0
            # Optimization: Pre-compute order? 
            # Since mask changes, we just scan.
            # To speed up, we can pre-sort indices by weight and check if in mask.
            for idx in range(n):
                if (mask >> idx) & 1:
                    if weights[idx] > max_w:
                        max_w = weights[idx]
                        v = idx
            if v == -1:
                break
            S.append(v)
            score += weights[v]
            # Remove v and its neighbors
            mask &= ~(neighbor_masks[v] | (1 << v))
        return S, score
        
    # Get initial solution
    init_set, init_score = greedy_solve((1 << n) - 1)
    best_score = init_score
    best_set = init_set
    
    # Local Search to improve initial solution
    def local_search(S):
        S_set = set(S)
        current_score = sum(weights[i] for i in S)
        candidates = [i for i in range(n) if i not in S_set]
        
        for _ in range(100): # Max iterations
            if time.time() > deadline - 5:
                break
            improved = False
            # Try add
            best_add_v = -1
            best_add_w = -1.0
            for v in candidates:
                if weights[v] > best_add_w:
                    # Check compatibility
                    comp = True
                    for u in S_set:
                        if conflict[v][u]:
                            comp = False
                            break
                    if comp:
                        best_add_w = weights[v]
                        best_add_v = v
            if best_add_v != -1:
                S_set.add(best_add_v)
                S.append(best_add_v)
                candidates.remove(best_add_v)
                current_score += best_add_w
                improved = True
                continue
                
            # Try swap
            best_swap_u = -1
            best_swap_v = -1
            best_gain = 1e-9
            for u in list(S_set):
                best_v = -1
                best_v_w = -1.0
                for v in candidates:
                    if weights[v] <= best_v_w:
                        continue
                    comp = True
                    for w in S_set:
                        if w == u:
                            continue
                        if conflict[v][w]:
                            comp = False
                            break
                    if comp:
                        best_v_w = weights[v]
                        best_v = v
                if best_v != -1:
                    gain = weights[best_v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap_u = u
                        best_swap_v = best_v
            if best_swap_u != -1 and best_swap_v != -1:
                S_set.remove(best_swap_u)
                S_set.add(best_swap_v)
                S[S.index(best_swap_u)] = best_swap_v
                candidates.remove(best_swap_v)
                candidates.append(best_swap_u)
                current_score += best_gain
                improved = True
                continue
                
            if not improved:
                break
        return S, current_score
        
    best_set, best_score = local_search(best_set)
    
    # Branch and Bound
    # We use a stack for DFS.
    # State: (available_mask, current_score, current_set)
    # To save memory, we don't store current_set, just score and mask.
    # But we need to reconstruct the set if we find a better one.
    # We can store the path or just the set.
    # Given N=255, storing sets is expensive.
    # Let's store the list of chosen vertices.
    
    # To make B&B effective, we need a good Upper Bound.
    # UB = current_score + sum(weights[i] for i in available_mask)
    # This is valid.
    
    # Optimization:
    # Precompute prefix sums of weights sorted by index? No, weights are fixed.
    # We can precompute a list of (index, weight) sorted by weight descending.
    # Then for a given mask, the sum of weights is just summing weights of bits in mask.
    # Python int bit_count is fast, but summing weights requires iteration.
    
    # Let's try to run B&B with a time limit of 5 seconds.
    # If it finishes, we have the global optimum.
    # If not, we return the best found so far.
    
    bbb_start = time.time()
    bbb_time_limit = 5.0
    
    # Stack: (mask, score, chosen_list)
    # We start with all vertices available.
    # But we already have a good solution.
    # We can prune any branch where score + UB <= best_score.
    
    # To speed up "sum of weights in mask":
    # We can precompute weights array.
    # Iterating through set bits of mask:
    # while m:
    #   lsb = m & -m
    #   idx = lsb.bit_length() - 1
    #   sum += weights[idx]
    #   m ^= lsb
    
    # This is O(k) where k is number of set bits.
    
    # Let's implement a recursive B&B with depth limit or time check.
    # Python recursion depth limit is ~1000. N=255 is fine.
    
    sys.setrecursionlimit(3000)
    
    def upper_bound(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s
        
    def branch_and_bound(mask, score, chosen):
        nonlocal best_score, best_set
        
        if time.time() > bbb_start + bbb_time_limit:
            return
            
        if mask == 0:
            if score > best_score:
                best_score = score
                best_set = chosen[:]
            return
            
        # Prune
        ub = score + upper_bound(mask)
        if ub <= best_score:
            return
            
        # Select vertex: Highest weight in mask
        v = -1
        max_w = -1.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        # Option 1: Include v
        # New mask: remove v and neighbors
        new_mask_incl = mask & ~(neighbor_masks[v] | (1 << v))
        branch_and_bound(new_mask_incl, score + weights[v], chosen + [v])
        
        # Option 2: Exclude v
        new_mask_excl = mask & ~(1 << v)
        branch_and_bound(new_mask_excl, score, chosen)
        
    # Start B&B
    # We can start from the empty set, but we have a good LB.
    # To make it faster, we could start from the greedy solution's complement?
    # No, standard B&B starts from root.
    
    # However, B&B on 255 nodes might be too slow even with 5s limit if the bound is weak.
    # Let's try it. If it's too slow, the time limit will catch it.
    
    # One optimization: If the number of available vertices is small, solve exactly?
    # No, B&B is exact.
    
    # Let's run it.
    try:
        branch_and_bound((1 << n) - 1, 0.0, [])
    except RecursionError:
        pass # Should not happen with limit 3000 and depth 255
    
    # If B&B didn't find anything better, we still have the local search result.
    # But wait, B&B explores the whole space. If it finishes, best_set is optimal.
    # If it times out, best_set is the best found during B&B or the initial one.
    
    # Return the values
    return [values[i] for i in best_set]

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
