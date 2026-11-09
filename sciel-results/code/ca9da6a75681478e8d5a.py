import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    candidates = list(range(2, n + 1))
    m = n - 1  # 255 elements indexed 0..254, element i corresponds to value i+2
    
    # Build conflict structure: for each element i, which pairs (j,k) with j<k, j!=i, k!=i
    # create a conflict if (i+2) divides (j+2)*(k+2)
    # Actually, the constraint is: for distinct a,b,c in set, a does NOT divide b*c
    # So a conflict exists for triple {a,b,c} if a|bc OR b|ac OR c|ab
    # For branch and bound, I'll track: for each element i, the set of pairs (j,k) that conflict with i
    # i.e., including i along with both j and k creates a violation
    
    # Precompute: for each i, conflict_pairs[i] = set of (j,k) with j<k, j!=i, k!=i such that
    # (i+2) | (j+2)*(k+2) or (j+2) | (i+2)*(k+2) or (k+2) | (i+2)*(j+2)
    
    # This is expensive. Let me think differently.
    # For branch and bound, I need to check if adding element i to current set S is valid.
    # That means: for all pairs (j,k) in S with j!=k, j!=i, k!=i, the triple {i,j,k} must be valid.
    # Triple {i,j,k} is valid iff NOT (i|jk or j|ik or k|ij) where i,j,k are values.
    
    # Let me precompute for each pair (j,k) the set of elements i that conflict with this pair.
    # conflict_with_pair[j][k] = set of i such that triple {i,j,k} is invalid.
    
    # Actually, let me precompute: for each i, the set of pairs (j,k) that are incompatible with i.
    # incompat[i] = set of frozenset({j,k}) where {i,j,k} is an invalid triple.
    
    # For efficiency, let me use a different approach:
    # Precompute a 3D boolean or use bitsets.
    
    # Let me precompute: conflict[i] is a set of pairs. When I have current set S and want to add i,
    # I check if any pair (j,k) in S is in conflict[i].
    
    # Precompute conflict[i] as a set of encoded pairs (j*m + k) for j<k
    conflict = [set() for _ in range(m)]
    
    for i in range(m):
        vi = i + 2
        for j in range(m):
            if j == i:
                continue
            vj = j + 2
            for k in range(j + 1, m):
                if k == i:
                    continue
                vk = k + 2
                # Check if triple {vi, vj, vk} is invalid
                # Invalid if vi | vj*vk or vj | vi*vk or vk | vi*vj
                if (vj * vk) % vi == 0 or (vi * vk) % vj == 0 or (vi * vj) % vk == 0:
                    # This pair (j,k) conflicts with i
                    conflict[i].add(j * m + k)
    
    # Now branch and bound
    # State: current set (as list), available elements (as set or list)
    # Upper bound: |current| + |available| (trivial)
    # Better upper bound: for each available element, count how many other available elements it's compatible with
    # Use greedy coloring bound or degree-based bound
    
    best_set = []
    best_size = 0
    
    # Use bitsets for efficiency
    # available_mask: integer with bits set for available elements
    # current_set: list of elements in current solution
    
    def is_valid_addition(current_set, i):
        """Check if adding i to current_set is valid (no conflicts with any pair in current_set)"""
        ci = conflict[i]
        for j in current_set:
            if j == i:
                continue
            for k in current_set:
                if k == j or k == i:
                    continue
                # Check if pair (j,k) is in conflict[i]
                if j < k:
                    if j * m + k in ci:
                        return False
                else:
                    if k * m + j in ci:
                        return False
        return True
    
    # This is O(|S|^2) per check. For branch and bound, this might be slow.
    # Let me optimize: precompute for each i, a bitset of compatible pairs.
    # Actually, let me use a different approach for the bound.
    
    # Upper bound using greedy independent set on the "pair conflict graph"
    # When we have current set S and available set A, the max we can add is bounded by
    # the size of a maximum independent set in the graph where vertices are A and edges
    # represent pairs that conflict with some element in S... no, that's not right.
    
    # Simpler upper bound: |S| + |A| (trivial)
    # Better: use the fact that each element in A can conflict with at most some number of others.
    
    # Let me use a simple branch and bound with the trivial bound first, and see if it's fast enough.
    
    # Actually, let me think about this problem differently.
    # The key observation: for prime p, if p is in the set, no multiple of p can be in the set.
    # So the problem has a strong structure.
    
    # Let me try a different approach: enumerate which primes to include, then determine
    # which composites can be included.
    
    # Primes in 2..256:
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    composites = [i for i in range(2, n + 1) if not is_prime[i]]
    
    # For a given set of primes P, which composites can we include?
    # A composite c can be included iff for every prime p in P, p does not divide c.
    # Wait, that's not quite right. The constraint is about triples.
    # If p is in the set and c is in the set, and there's another element d in the set,
    # then we need p | c*d to be false, c | p*d to be false, d | p*c to be false.
    
    # This is complex. Let me just do the branch and bound on the full problem.
    
    # Let me use a smarter approach: start with all primes, then try to swap.
    # Or use the hypergraph MIS approach properly.
    
    # Let me implement a proper branch and bound with a good upper bound.
    # Upper bound: for the available set A, compute a greedy coloring of the "conflict graph"
    # where two elements are adjacent if they conflict with each other (i.e., there exists
    # a pair in the current set such that both conflict with that pair... no).
    
    # Actually, the constraint is: we can't have three elements forming an invalid triple.
    # This is a 3-uniform hypergraph independent set.
    
    # For upper bound, I'll use: |S| + maximum matching in the hypergraph restricted to A.
    # Or simpler: |S| + |A| / 1 (trivial).
    
    # Let me just try a heuristic approach: start with primes, try to add composites by removing conflicts.
    
    # Better: let me do a local search starting from the all-primes set.
    
    S = set(primes)
    best_S = set(primes)
    best_size = len(primes)
    
    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(len(S_list)):
                if i == j:
                    continue
                b = S_list[j]
                for k in range(len(S_list)):
                    if i == k or j == k:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Local search: try to add composites, removing conflicting elements if needed
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding each composite
        for c in composites:
            if c in S:
                continue
            if time.time() > deadline - 5:
                break
            
            # Find all elements in S that conflict with c (i.e., for some pair (x,y) in S,
            # the triple {c,x,y} is invalid)
            # An element x in S "blocks" c if there exists y in S, y!=x, such that
            # triple {c,x,y} is invalid.
            
            # To add c, we need to remove a set R from S such that S\R ∪ {c} is valid.
            # Minimum R: remove all x such that x is in some invalid triple with c and another element.
            
            # For each x in S, check if there exists y in S, y!=x, y!=c, such that
            # triple {c,x,y} is invalid.
            # If yes, x must be removed (or all such x's).
            
            must_remove = set()
            for x in S:
                if x == c:
                    continue
                for y in S:
                    if y == x or y == c:
                        continue
                    # Check if {c,x,y} is invalid
                    if (x * y) % c == 0 or (c * y) % x == 0 or (c * x) % y == 0:
                        must_remove.add(x)
                        break
            
            # After removing must_remove, check if S \ must_remove ∪ {c} is valid
            new_S = S - must_remove
            new_S.add(c)
            
            if is_valid_set(new_S) and len(new_S) > best_size:
                S = new_S
                best_S = set(new_S)
                best_size = len(new_S)
                improved = True
    
    # Also try removing primes to add multiple composites
    # For each prime p, try removing it and see if we can add more composites
    if time.time() < deadline - 5:
        for p in list(S):
            if not is_prime[p]:
                continue
            S_without_p = S - {p}
            # Try to add composites
            for c in composites:
                if c in S_without_p:
                    continue
                must_remove = set()
                for x in S_without_p:
                    if x == c:
                        continue
                    for y in S_without_p:
                        if y == x or y == c:
                            continue
                        if (x * y) % c == 0 or (c * y) % x == 0 or (c * x) % y == 0:
                            must_remove.add(x)
                            break
                new_S = S_without_p - must_remove
                new_S.add(c)
                if is_valid_set(new_S) and len(new_S) > best_size:
                    S = new_S
                    best_S = set(new_S)
                    best_size = len(new_S)
    
    # Convert to list of ints
    result = [int(x) for x in best_S]
    result.sort()
    
    # Verify
    if not is_valid_set(set(result)):
        # Fallback to primes
        result = [int(x) for x in primes]
    
    return result
