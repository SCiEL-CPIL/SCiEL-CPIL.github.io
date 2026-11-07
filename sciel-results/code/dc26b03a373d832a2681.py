"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

MAX_N = 256
DEADLINE = 85.0  # Safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    
    N = MAX_N  # 256
    # Map numbers 2..256 to indices 0..254
    nums = list(range(2, N + 1))
    num_to_idx = {nums[i]: i for i in range(len(nums))}
    n = len(nums)  # 255 elements
    
    # Precompute prime factors
    pf = {}
    for x in range(2, N + 1):
        pf[x] = _prime_factors(x)
    
    # Build conflict structure:
    # For each vertex v, compute the set of other vertices that form a conflict with v.
    # A conflict triple is (a, b, c) distinct where a | b*c.
    # For maximum independent set in 3-uniform hypergraph:
    # We need: no three distinct vertices a,b,c in our set such that a|b*c.
    # 
    # Key insight: If we fix which two vertices are "the product" (b,c), then any a that divides b*c
    # cannot coexist with both b and c.
    #
    # Alternative approach: Build a pairwise "conflict" matrix where conflict[i][j] = set of k
    # such that adding k would create a violation with i and j.
    # Actually, let's think differently.
    #
    # For a set S to be valid: for all distinct a,b,c in S, NOT (a | b*c).
    # 
    # Let's precompute for each pair (i,j) (i != j), the set of k (k != i, k != j) such that
    # nums[k] | nums[i]*nums[j] OR nums[i] | nums[j]*nums[k] OR nums[j] | nums[i]*nums[k].
    #
    # Actually, simpler: for each ordered triple of distinct indices (a_idx, b_idx, c_idx),
    # check if nums[a_idx] | nums[b_idx]*nums[c_idx]. If yes, then {a_idx, b_idx, c_idx} is a bad triple.
    #
    # For branch-and-bound on maximum independent set in hypergraph:
    # We maintain a set of candidate vertices. A vertex v can be added to current solution S
    # only if for every pair (u,w) in S (u!=w, u!=v, w!=v), v does not form a bad triple with u and w.
    # Also, for every u in S, and every w in S with w != u, v != u, v != w:
    #   nums[v] | nums[u]*nums[w] must be false,
    #   nums[u] | nums[v]*nums[w] must be false,
    #   nums[w] | nums[v]*nums[u] must be false.
    #
    # Let's precompute: for each vertex v, and for each pair (u,w) with u<w, u!=v, w!=v,
    # whether v conflicts with the pair (u,w).
    #
    # Actually, let's use a different formulation. For each vertex v, define:
    #   conflict_with_pair[v][u][w] = True if {v,u,w} is a bad triple
    #
    # This is 255 * 255 * 255 = too much memory.
    #
    # Better: For each vertex v, precompute the set of pairs (u,w) such that {v,u,w} is bad.
    # Then when checking if v can be added to S, we check if any pair in S is in this set.
    #
    # Even better: For each vertex v, precompute a bitset of vertices that "conflict" with v.
    # But the conflict is 3-way, not 2-way.
    #
    # Let me think about this more carefully. The condition is:
    # For distinct a, b, c: a does NOT divide b*c.
    #
    # This is equivalent to: for each a, the set of pairs (b,c) from S\{a} such that a|b*c must be empty.
    #
    # So for each a, define F(a) = {(b,c) : b != c, b != a, c != a, a | b*c}
    # A set S is valid iff for all a in S, S\{a} contains no pair from F(a).
    #
    # For branch-and-bound:
    # State: current solution S, remaining candidates R.
    # Upper bound: |S| + |R| (trivial) or better: |S| + max independent set in the induced subgraph.
    #
    # Let's use a simpler approach: greedy + local search with bitsets.
    
    # Precompute: for each vertex i, a bitset (as Python int) of vertices j such that
    # there exists k != i, k != j with nums[i] | nums[j]*nums[k] or similar.
    # Actually, let's precompute pairwise "incompatibility" in a different way.
    
    # For each pair (i, j) with i < j, compute the set of vertices k (k != i, k != j)
    # such that {i, j, k} is a bad triple.
    # Then, if we have both i and j in our solution, we cannot have any k in that set.
    
    # Let's precompute: for each pair (i,j), a bitset of forbidden third vertices.
    # pair_forbidden[i][j] = bitset of k such that {i,j,k} is a bad triple
    
    pair_forbidden = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            forbidden = 0
            a = nums[i]
            b = nums[j]
            ab = a * b
            for k in range(n):
                if k == i or k == j:
                    continue
                c = nums[k]
                # Check if {a, b, c} is a bad triple:
                # a | b*c, or b | a*c, or c | a*b
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    forbidden |= (1 << k)
            pair_forbidden[i][j] = forbidden
            pair_forbidden[j][i] = forbidden
    
    # Now, for a set S to be valid: for every pair (i,j) in S, S must not contain any vertex
    # in pair_forbidden[i][j].
    #
    # Branch and bound:
    # We want to find the maximum subset S of {0,...,n-1} such that for every pair (i,j) in S,
    # S ∩ {k : k in pair_forbidden[i][j]} = empty.
    
    # Use a recursive branch-and-bound with bitsets.
    # State: available vertices (bitset), current solution size, current best.
    
    # Precompute degree for ordering: for each vertex, count how many other vertices it conflicts with.
    # Vertex i conflicts with vertex j if... it's not a simple pairwise conflict.
    # Let's use a heuristic: order vertices by their "degree" in the hypergraph,
    # i.e., the number of pairs (j,k) such that {i,j,k} is a bad triple.
    
    vertex_degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Count vertices k that form a bad triple with i and j
            cnt = bin(pair_forbidden[i][j]).count('1')
            vertex_degree[i] += cnt
            vertex_degree[j] += cnt
    
    # Sort vertices by degree (ascending - try low-degree vertices first as they're "safer")
    # Actually, for maximum independent set, we often want to branch on high-degree vertices first
    # to prune more. But for adding to solution, low-degree vertices are easier to include.
    # Let's try: in the branch, we decide include/exclude. For include, pick a vertex that
    # is compatible with the current solution.
    
    # Let's use a different approach: iterative greedy + local search with bitsets.
    
    def can_add_to(S_bitset, v):
        """Check if vertex v can be added to the set represented by S_bitset."""
        if S_bitset & (1 << v):
            return True  # already in set
        # For each u in S_bitset, check if v is in pair_forbidden[u][v]
        # i.e., check if there exists u in S such that bit v is set in pair_forbidden[u][v]
        # Actually, we need: for each u in S (u != v), v is NOT in pair_forbidden[u][v]
        # which means pair_forbidden[u][v] & (1 << v) == 0
        # But pair_forbidden[u][v] is a bitset of k such that {u,v,k} is bad.
        # So we need: for all u in S, u != v: (pair_forbidden[u][v] & (1 << v)) == 0
        # Wait, that's not right. pair_forbidden[u][v] contains k such that {u,v,k} is bad.
        # We want to check if v can be added. The condition is: for all pairs (u, w) in S ∪ {v},
        # no vertex in S ∪ {v} is in pair_forbidden[u][w].
        # Since S is already valid, we only need to check pairs involving v.
        # For each u in S (u != v): the pair (u, v) is new. We need S ∩ pair_forbidden[u][v] = empty.
        # Also, v is not in pair_forbidden[u][v] by construction (k != i, k != j).
        # So: for all u in S, u != v: (S_bitset & pair_forbidden[u][v]) == 0
        
        # Iterate over set bits in S_bitset
        temp = S_bitset
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            u_idx = lb.bit_length() - 1
            if u_idx != v:
                if S_bitset & pair_forbidden[u_idx][v]:
                    return False
            temp ^= lb
        return True
    
    # Greedy construction
    S_bitset = 0
    # Order vertices by degree (ascending)
    order = sorted(range(n), key=lambda x: vertex_degree[x])
    
    for v in order:
        if time.time() - start_time > DEADLINE - 10:
            break
        if can_add_to(S_bitset, v):
            S_bitset |= (1 << v)
    
    best_size = bin(S_bitset).count('1')
    best_bitset = S_bitset
    
    # Local search: try to add vertices that don't fit by removing one existing vertex
    # Try each vertex not in S, and for each, try removing each vertex in S and see if it fits.
    
    # First, let's try a simpler improvement: try adding vertices in different order
    # and also try random restarts.
    
    import random
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        random.seed(restart)
        S_bitset = 0
        
        # Random order with some bias
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            if time.time() - start_time > DEADLINE - 5:
                break
            if can_add_to(S_bitset, v):
                S_bitset |= (1 << v)
        
        sz = bin(S_bitset).count('1')
        if sz > best_size:
            best_size = sz
            best_bitset = S_bitset
    
    # Try local search: swap one vertex out, one in
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        improved = False
        # Get list of vertices in S
        S_list = []
        temp = best_bitset
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            S_list.append(idx)
            temp ^= lb
        
        # Get list of vertices not in S
        not_S = [i for i in range(n) if not (best_bitset & (1 << i))]
        
        random.shuffle(not_S)
        
        for v in not_S:
            if time.time() - start_time > DEADLINE - 3:
                break
            for u in S_list:
                if u == v:
                    continue
                # Try removing u and adding v
                new_bitset = (best_bitset & ~(1 << u)) | (1 << v)
                # Check validity: for each pair in new_bitset, no forbidden vertex is in new_bitset
                valid = True
                # We only need to check pairs involving v (since S\{u} was valid)
                temp2 = new_bitset & ~(1 << v)
                while temp2:
                    lb = temp2 & (-temp2)
                    w_idx = lb.bit_length() - 1
                    if w_idx != v:
                        if new_bitset & pair_forbidden[w_idx][v]:
                            valid = False
                            break
                    temp2 ^= lb
                if valid:
                    best_bitset = new_bitset
                    best_size = bin(best_bitset).count('1')
                    improved = True
                    break
            if improved:
                break
    
    # Convert best_bitset to list of integers
    result = []
    temp = best_bitset
    while temp:
        lb = temp & (-temp)
        idx = lb.bit_length() - 1
        result.append(nums[idx])
        temp ^= lb
    
    result.sort()
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
