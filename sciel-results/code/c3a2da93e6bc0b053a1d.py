import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # The set must contain 255.
    # Any other element x must have gcd(x, 255) > 1.
    # So x must be divisible by 3, 5, or 17.
    # Let's generate the candidate set C.
    candidates = []
    for i in range(1, 256):
        if i == 255:
            continue
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Add 255 to the list of nodes for the clique algorithm, 
    # but we know 255 is in the clique.
    # Actually, it's easier to just find the max clique in the graph of candidates,
    # and the final answer is that clique + [255].
    # Why? Because 255 is coprime to none of them? 
    # gcd(255, x) > 1 for all x in candidates.
    # So 255 forms a clique with any clique in candidates.
    # Thus, MaxClique(G[C] + 255) = MaxClique(G[C]) + 1.
    
    n = len(candidates)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of neighbors of candidates[i]
    adj = [0] * n
    
    # Precompute gcds? No, just compute on fly or precompute a matrix.
    # 127x127 is small.
    # Let's use bitmasks for speed.
    
    # Optimization: Group candidates by their prime factors?
    # No, just build the graph.
    
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Bron-Kerbosch with pivot
    # R: current clique
    # P: candidates to add
    # X: candidates already processed (to avoid duplicates)
    
    max_clique_size = 0
    best_clique = []
    
    # Use a recursive function
    # To avoid recursion depth issues or overhead, we can implement it iteratively or just use recursion with a small depth.
    # Depth is at most 127, which is fine.
    
    # Pivot selection: choose v in P union X that maximizes |P & adj[v]|
    # Then iterate over P \ adj[v]
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(P) == 0 and len(X) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                best_clique = R[:]
            return
        
        # If P is empty but X is not, we still found a maximal clique, but it's a subset of a previous one?
        # Standard BK: if P is empty, we have a maximal clique.
        # The condition for maximal clique is P is empty. X is for expansion.
        # Actually, the standard algorithm reports a clique when P is empty.
        # We want the maximum one.
        
        if len(P) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                best_clique = R[:]
            return
            
        # Choose pivot u from P union X
        # We want to minimize the number of recursive calls, so we pick u that maximizes the number of neighbors in P.
        # Then we only need to recurse on P \ N(u).
        
        # Find pivot
        # Convert P to list of indices for easier manipulation? Or use bitmasks.
        # Bitmasks are faster.
        
        # To find pivot, we can iterate through P and X.
        # Let's convert P and X to lists of indices if they are small, or stick to bitmasks.
        # n=127, so bitmasks are fine.
        
        # We need to iterate over candidates in P \ N(u).
        
        # First, find u.
        # max_count = -1
        # u = -1
        # for v in P union X:
        #    count = popcount(P & adj[v])
        #    if count > max_count:
        #        max_count = count
        #        u = v
        
        # To iterate P union X, we can combine bitmasks.
        # P and X are disjoint.
        # PX = P | X
        # Iterate through set bits in PX.
        
        # However, iterating through set bits in Python is slower than iterating a list.
        # Let's maintain P and X as lists of indices?
        # With n=127, lists are fine and might be faster than bitmask ops in Python due to overhead of bit manipulation functions.
        # Let's try lists.
        
        # Re-implementation with lists for P and X, and a set for R?
        # Or just lists for R, P, X.
        
        pass # Placeholder, will implement below

    # Let's stick to bitmasks for adjacency, but lists for P, X, R.
    # This is often faster in Python for moderate N.
    
    def bk(R, P, X):
        nonlocal max_clique_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                best_clique = R[:]
            return
        
        # Choose pivot u from P union X
        # We want u that maximizes |P ∩ N(u)|
        # Let's just pick the first one? Or the one with max degree in P?
        # Picking max degree in P is the standard heuristic.
        
        # To find max degree in P:
        # For each v in P union X:
        #    count = number of neighbors of v in P
        
        # Since P and X are lists, we can iterate.
        # But checking "is neighbor" requires bitmask lookup or a set.
        # adj[v] is a bitmask.
        # To count neighbors in P (list), we can do:
        # sum(1 for p in P if (adj[v] >> p) & 1)
        
        # This is O(|P| * |P ∪ X|) which is O(N^3) worst case per node? No, O(N^2) per node.
        # Total complexity might be high.
        
        # Alternative: Precompute adjacency as a list of sets?
        # adj_set[i] = set of neighbors.
        
        # Let's rebuild adjacency as sets for easier intersection.
        
    # Let's restart the inner logic with sets.
    
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_set[i].add(j)
                adj_set[j].add(i)
                
    # Initial P is all nodes, X is empty, R is empty.
    # But we can prune.
    # Since we know the clique must be large, and multiples of 3 form a clique of size 85,
    # we can start with that as a lower bound.
    
    # Let's just run BK.
    
    def bk_sets(R, P, X):
        nonlocal max_clique_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                best_clique = R[:]
            return
            
        # Choose pivot u from P ∪ X
        # u should maximize |P ∩ N(u)|
        # To minimize recursion, we iterate over P \ N(u).
        
        # Find u
        # If P ∪ X is empty, we would have returned.
        # We can just pick u = P[0] for simplicity?
        # Picking the one with max degree in P is better.
        
        # Let's find u
        u = -1
        max_neighbors = -1
        # We can iterate over P ∪ X
        # Since X is usually small or P is shrinking, let's just iterate P.
        # Actually, the pivot can be in X too.
        # But picking from P is usually sufficient and faster to check.
        
        # Let's just pick the vertex in P with the highest degree in P.
        # This is a common heuristic.
        
        for v in P:
            deg = len(adj_set[v] & P)
            if deg > max_neighbors:
                max_neighbors = deg
                u = v
                
        # Now iterate over P \ N(u)
        # N_u = adj_set[u]
        # Candidates = P - N_u
        
        # We need to remove candidates from P and add to X as we go?
        # Standard BK:
        # for v in P \ N(u):
        #    bk(R ∪ {v}, P ∩ N(v), X ∩ N(v))
        #    P.remove(v)
        #    X.add(v)
        
        # To avoid modifying P while iterating, make a copy or list of candidates.
        # candidates_to_visit = list(P - adj_set[u])
        
        # Note: P is a set.
        
        for v in list(P - adj_set[u]):
            if time.time() > deadline:
                return
            
            # New P is P ∩ N(v)
            new_P = P & adj_set[v]
            # New X is X ∩ N(v)
            new_X = X & adj_set[v]
            
            bk_sets(R | {v}, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
    # Initial call
    # R = set()
    # P = set(range(n))
    # X = set()
    
    # Optimization: We can start with a known clique to set initial best.
    # Multiples of 3:
    mult3 = [i for i in range(n) if candidates[i] % 3 == 0]
    # This is a clique.
    max_clique_size = len(mult3)
    best_clique = mult3[:]
    
    # Run BK
    bk_sets(set(), set(range(n)), set())
    
    # The best clique found is in indices of 'candidates'.
    # We need to map back to values.
    result = [candidates[i] for i in best_clique]
    result.append(255)
    
    return result

# Test locally if possible?
# The logic seems sound.
# One issue: recursion depth. Max 127. Python default is 1000. Safe.
# Time complexity: BK with pivot is fast for dense graphs.
# Is the graph dense?
# Nodes are numbers sharing factors 3, 5, 17.
# Multiples of 3 are connected to each other.
# Multiples of 5 are connected to each other.
# Multiples of 17 are connected to each other.
# Connections between groups:
# 3 and 5: multiples of 15.
# 3 and 17: multiples of 51.
# 5 and 17: multiples of 85.
# It's quite dense.
# The clique of all multiples of 3 has size 85.
# Can we add multiples of 5?
# A multiple of 5 (not 3) is coprime to a multiple of 3 (not 5) unless they share another factor.
# e.g., 5 and 3 are coprime.
# So we can't just take all multiples of 3 and all multiples of 5.
# We must pick a subset.
# The max clique will likely be close to 85 or slightly larger if we can swap some 3s for 5s/17s that connect well.
# Actually, since 3 is the most common factor, the clique of 3s is a strong baseline.
# Let's see if we can improve.
# If we take all multiples of 3, we have 85 nodes.
# Can we add any multiple of 5 (not 3)?
# Let x be a multiple of 5, not 3.
# For x to be in the clique, it must be connected to ALL multiples of 3 in the clique.
# gcd(x, y) > 1 for all y in clique.
# If y is a multiple of 3, and x is not, they share a factor only if x has another factor in common with y.
# This is hard to satisfy for ALL y.
# Example: y=3. gcd(x,3)>1 implies 3|x. But x is not multiple of 3. Contradiction.
# So if 3 is in the clique, NO other number coprime to 3 can be in the clique.
# Since 3 is a multiple of 3, if we include 3, we can only include numbers divisible by 3.
# So the clique of all multiples of 3 is maximal with respect to adding non-multiples of 3?
# Wait. If we remove 3 from the clique, we might be able to add numbers not divisible by 3.
# But we lose 3.
# Is it better to have a clique of size 85 (all mult of 3) or a smaller clique of mult of 3 plus some mult of 5?
# If we remove 3, we can potentially add 5.
# But 5 is not connected to 9 (gcd 1).
# So we would have to remove 9 too.
# It seems the "all multiples of 3" strategy is very robust.
# The algorithm will find the true maximum.

# One detail: "Return a nonempty subset...".
# The return type is a list.

# Final check on constraints:
# "Maximize cardinality"
# "gcd(a,b)>1 for all distinct pairs"
# "containing 255"

# The code looks correct.

# One potential issue: `bk_sets` uses `P` and `X` as sets.
# `P & adj_set[v]` creates a new set.
# `P.remove(v)` modifies P.
# This is correct for BK.

# Let's double check the pivot logic.
# Standard BK with pivot:
# if P is empty:
#   report R
# else:
#   choose u in P union X
#   for v in P \ N(u):
#     bk(R U {v}, P ∩ N(v), X ∩ N(v))
#     P = P \ {v}
#     X = X U {v}

# My code does exactly this.
# The choice of u: I chose u in P with max degree in P.
# This is a valid heuristic.

# Time limit: 85s.
# For N=127, this should be instantaneous.

# What if the max clique is much larger?
# Max possible size is 127 (if all connected).
# But they are not all connected.
# The graph is not a clique.
# So BK will prune heavily.

# Let's write the code.
