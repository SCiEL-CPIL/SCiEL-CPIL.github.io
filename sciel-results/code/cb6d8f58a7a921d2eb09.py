import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                if gcd_table[b][c] == g_ab and gcd_table[c][a] == g_ab:
                    bad_triples.append((a, b, c))
    
    # For each element x, precompute a bitmask of elements that form a bad triple with x
    # conflict_mask[x] has bit y set if there exists z such that (x,y,z) is a bad triple
    # But we need more: for branch and bound on hypergraph independent set,
    # we need to know which elements are "conflicting" with x in the sense that
    # adding x would create a bad triple.
    
    # Actually, for a 3-uniform hypergraph, an independent set is a set with no bad triple.
    # We can model this as: for each pair (a,b), if both are in the set, then no x in bad_pair_mask[(a,b)] can be added.
    
    # Better approach: use a different formulation.
    # For each element x, define its "conflict neighborhood" as the set of elements y such that
    # there exists z where (x,y,z) is bad. But this isn't quite right for hypergraphs.
    
    # Let's use a different strategy: local search with better move generation.
    # Or use a genetic algorithm approach.
    
    # Actually, let's try a different structural change: use a constraint-based approach
    # where we build the set incrementally with backtracking and pruning.
    
    # Precompute for each x: list of pairs (a,b) such that (x,a,b) is bad
    x_bad_pairs = [[] for _ in range(N+1)]
    for (a, b, c) in bad_triples:
        x_bad_pairs[a].append((b, c))
        x_bad_pairs[b].append((a, c))
        x_bad_pairs[c].append((a, b))
    
    def count_conflicts(S_mask, x):
        """Count number of bad triples formed by adding x to set S_mask."""
        count = 0
        for (a, b) in x_bad_pairs[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                count += 1
        return count
    
    def popcount(mask):
        return bin(mask).count('1')
    
    # Branch and bound: try to build a large independent set
    # State: current set S, remaining elements R, best size found so far
    # We use a greedy heuristic to get a lower bound, then branch.
    
    def greedy_lower_bound(available_mask):
        """Get a lower bound by greedily adding elements."""
        S = 0
        remaining = available_mask
        while remaining:
            # Try to find an element that can be added
            # Use a simple heuristic: try elements in order
            for x in range(1, N+1):
                if remaining & (1 << x):
                    if count_conflicts(S, x) == 0:
                        S |= (1 << x)
                        remaining &= ~(1 << x)
                        break
            else:
                break
        return popcount(S), S
    
    best_size = 0
    best_S = 0
    
    # First, get a good initial solution using greedy
    init_size, init_S = greedy_lower_bound(FULL_MASK)
    best_size = init_size
    best_S = init_S
    
    # Now use branch and bound
    # We'll use a recursive approach with memoization of visited states
    # But the state space is too large, so we'll use a heuristic-guided search.
    
    # Alternative: use a local search with tabu list
    # Or use a different approach: formulate as a maximum clique problem in the complement graph.
    
    # Let's try a different structural change: use a simulated annealing with better
    # move generation and a different cooling schedule.
    
    # Actually, let's implement a proper branch and bound with bitset operations.
    # Key insight: for a 3-uniform hypergraph, we can use the following approach:
    # 1. Pick an element x.
    # 2. Either include x, in which case we must exclude all elements that form a bad triple with x.
    #    But "exclude" is not straightforward because the constraint is on triples, not pairs.
    
    # Let's use a different formulation: for each pair (a,b) in the set, we know which elements
    # cannot be added. So we can maintain a "forbidden" set.
    
    # State: (S, forbidden) where S is the current set and forbidden is the set of elements
    # that cannot be added because they would form a bad triple with some pair in S.
    
    # When we add x to S, we update forbidden: for each pair (a,b) in S, if x is in bad_pair_mask[(a,b)],
    # then... wait, that's not right. When we add x, we need to add to forbidden all elements y such that
    # there exists a pair (a,b) in S where (x,y,z) is bad for some z. Actually, it's simpler:
    # When we add x to S, the new forbidden elements are those y such that there exists a pair (a,b) in S
    # where (x,a,b) is bad... no, that's not right either.
    
    # Let me think more carefully. The constraint is: no three distinct elements x,y,z in the set
    # should have gcd(x,y) = gcd(x,z) = gcd(y,z).
    
    # So if we have a current set S and we want to add x, we need to check that for all pairs (a,b) in S,
    # (x,a,b) is not a bad triple.
    
    # For branch and bound, we can use the following approach:
    # - Maintain the current set S.
    # - Maintain a list of "conflict pairs": for each pair (a,b) in S, the set of elements that cannot be added.
    # - When adding x, update the conflict pairs.
    
    # This is getting complex. Let's try a simpler approach: use a genetic algorithm or
    # a different local search strategy.
    
    # Actually, let's implement a proper maximum independent set solver using the following approach:
    # 1. Convert the 3-uniform hypergraph to a graph where two vertices are connected if they
    #    "conflict" in some sense. But this doesn't directly work for hypergraphs.
    
    # Let's try a different approach: use a constraint programming style search.
    # Variables: which elements are in the set.
    # Constraints: for each bad triple (a,b,c), not all three can be in the set.
    
    # We can use a backtracking search with the following pruning:
    # - If the current set size + number of remaining elements < best_size, prune.
    # - If adding any remaining element would create a bad triple, prune.
    
    # But this is still exponential in the worst case.
    
    # Let's try a hybrid approach: use local search to get a good initial solution,
    # then use branch and bound to try to improve it.
    
    # For now, let's implement a better local search with more aggressive move generation.
    
    def local_search(S, max_steps=10000):
        """Improve S using local search."""
        current = S
        current_conf = 0
        for (a, b, c) in bad_triples:
            if (current & (1 << a)) and (current & (1 << b)) and (current & (1 << c)):
                current_conf += 1
        
        improved = True
        steps = 0
        while improved and steps < max_steps and time.time() < deadline - 5.0:
            improved = False
            steps += 1
            
            # Try to add any element that doesn't create conflicts
            for x in range(1, N+1):
                if not (current & (1 << x)):
                    if count_conflicts(current, x) == 0:
                        current |= (1 << x)
                        improved = True
                        break
            
            # Try to remove one element and add two others
            if not improved:
                active = [i for i in range(1, N+1) if current & (1 << i)]
                inactive = [i for i in range(1, N+1) if not (current & (1 << i))]
                
                if len(active) >= 1 and len(inactive) >= 2:
                    # Try all single removals
                    for x_remove in active:
                        S_temp = current & ~(1 << x_remove)
                        # Try to add two elements
                        added = 0
                        for x_add in inactive:
                            if S_temp & (1 << x_add):
                                continue
                            if count_conflicts(S_temp, x_add) == 0:
                                S_temp |= (1 << x_add)
                                added += 1
                                if added == 2:
                                    break
                        
                        if added == 2:
                            current = S_temp
                            improved = True
                            break
        
        return current
    
    # Run local search from the greedy solution
    best_S = local_search(best_S)
    best_size = popcount(best_S)
    
    # Try multiple random restarts
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        S = 0
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if count_conflicts(S, x) == 0:
                S |= (1 << x)
        
        S = local_search(S, max_steps=5000)
        size = popcount(S)
        
        if size > best_size:
            best_size = size
            best_S = S
    
    # Final local search
    best_S = local_search(best_S, max_steps=20000)
    best_size = popcount(best_S)
    
    # Convert bitmask to list
    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    if not result:
        result = [1]
    return result
