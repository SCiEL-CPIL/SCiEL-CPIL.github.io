import time
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def is_valid(S):
        S_set = set(S)
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                gij = gcd_mat[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to S without creating a forbidden triple."""
        for i in range(len(S)):
            a = S[i]
            gax = gcd_mat[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True

    # Build the hypergraph: for each pair (a,b), find all c such that
    # gcd(a,b) == gcd(a,c) == gcd(b,c)
    # A triple (a,b,c) is forbidden if all three pairwise gcds are equal.
    # For branch and bound, we need: given a set S, which remaining vertices
    # are individually compatible?
    
    # Precompute for each x, the set of pairs (a,b) in any candidate set that
    # would conflict. Instead, let's precompute: for each x and each pair (a,b),
    # whether adding x to {a,b} creates a forbidden triple.
    # That is: gcd(a,b) == gcd(a,x) == gcd(b,x)
    
    # For efficiency in B&B, precompute for each x:
    # conflict_pairs[x] = set of (a,b) pairs (a<b, a,b in [1,128]) such that
    # gcd(a,b) == gcd(a,x) == gcd(b,x)
    # This is too large (128*127/2 * 128). Let's think differently.
    
    # Alternative: For B&B, at each node we have:
    # - chosen: set of selected vertices
    # - remaining: set of unprocessed vertices
    # Upper bound: len(chosen) + number of vertices in remaining that are
    #   individually compatible with chosen (can_add check)
    # This is a valid upper bound because any valid extension can only add
    # vertices that are individually compatible with chosen.
    
    # To make this fast, precompute for each x a bitmask of which other vertices
    # it's "compatible" with in the sense that no pair among them conflicts.
    # Actually, the compatibility is not pairwise - it's about triples.
    # x is compatible with set S iff for all pairs (a,b) in S, 
    # NOT (gcd(a,b)==gcd(a,x)==gcd(b,x)).
    
    # Let's precompute: for each x, for each pair (a,b), whether x conflicts
    # with that pair. Store as a set of frozensets or similar.
    # Better: for each x, store the set of (a,b) pairs that x conflicts with.
    
    # Actually for the B&B upper bound, I need: given chosen set C, how many
    # vertices in remaining are compatible with C? A vertex x is compatible
    # with C iff for all pairs (a,b) in C, gcd(a,b) != gcd(a,x) or gcd(a,b) != gcd(b,x).
    # Equivalently, x is NOT compatible iff there exists a pair (a,b) in C such that
    # gcd(a,b) == gcd(a,x) == gcd(b,x).
    
    # Precompute: for each x, the set of pairs (a,b) (a<b, both in [1,128]) 
    # such that gcd(a,b)==gcd(a,x)==gcd(b,x). Call this conflict_pairs_of[x].
    # Then x is incompatible with C iff conflict_pairs_of[x] intersects with
    # the set of pairs in C.
    
    # This is still potentially expensive. Let's try a different upper bound:
    # Just use len(chosen) + len(remaining) as a loose bound, and rely on
    # good vertex ordering to prune early.
    
    # Actually, let me try a smarter approach:
    # Order vertices by "conflict degree" - how many other vertices they
    # participate in forbidden triples with.
    
    # Count for each vertex x: how many pairs (a,b) with a,b != x such that
    # (a,b,x) is a forbidden triple.
    conflict_count = [0] * (N + 1)
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    conflict_count[x] += 1
    
    # Order vertices by decreasing conflict count (most constrained first)
    vertex_order = sorted(UNIVERSE, key=lambda x: -conflict_count[x])
    
    # For the B&B, we process vertices in this order.
    # State: index in vertex_order, chosen set, 
    # We decide for each vertex: include or exclude.
    
    # Upper bound at node (idx, chosen): 
    #   len(chosen) + (number of vertices from idx to end that are compatible with chosen)
    # For speed, I'll compute this by checking can_add for each remaining vertex.
    
    # To speed up can_add, precompute for each x the set of "witness pairs"
    # Actually, let's just use the direct check but memoize.
    
    # Let me try a different formulation that's faster:
    # For each x, precompute conflict_with[x] = set of y such that there exists
    # z where (x,y,z) is a forbidden triple. This isn't quite right either.
    
    # Let me just implement B&B with the straightforward upper bound and hope
    # the vertex ordering helps.
    
    best_solution = []
    best_len = 0
    
    # First, get a good initial solution via greedy
    def greedy_order(order):
        if order == "conflict_desc":
            return sorted(UNIVERSE, key=lambda x: -conflict_count[x])
        elif order == "conflict_asc":
            return sorted(UNIVERSE, key=lambda x: conflict_count[x])
        elif order == "natural":
            return list(UNIVERSE)
        elif order == "reverse":
            return list(reversed(UNIVERSE))
        else:
            return list(UNIVERSE)
    
    # Multiple greedy starts
    initial_solutions = []
    for order in ["conflict_asc", "conflict_desc", "natural", "reverse"]:
        selected = []
        S_set = set()
        for x in greedy_order(order):
            if can_add(selected, x):
                selected.append(x)
                S_set.add(x)
        initial_solutions.append(selected)
    
    best_solution = max(initial_solutions, key=len)
    best_len = len(best_solution)
    
    if time.time() >= DEADLINE - 5:
        return best_solution
    
    # Branch and bound
    # We'll use an iterative deepening / priority queue approach
    # Actually, let's do a recursive B&B with a time check.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For the B&B, we need an efficient "is x compatible with set C" check.
    # Precompute: for each x, a set of (a,b) pairs that x conflicts with.
    # Then x is incompatible with C iff any pair in C is in conflict_pairs_of[x].
    
    # Let's precompute conflict_pairs for each x as a set of encoded pairs.
    # Encode pair (a,b) as a*b + b (or a*(N+1)+b)
    
    conflict_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    # x conflicts with pair (a,b)
                    code = a * (N + 1) + b
                    conflict_pairs[x].add(code)
    
    def is_compatible(x, chosen_set):
        """Check if x is compatible with all pairs in chosen_set."""
        # x is incompatible iff there exists a pair (a,b) in chosen_set
        # such that (a,b) is in conflict_pairs[x]
        for a in chosen_set:
            for b in chosen_set:
                if a >= b:
                    continue
                code = a * (N + 1) + b
                if code in conflict_pairs[x]:
                    return False
        return True
    
    # This is still O(|C|^2) per check. For a set of size ~50, that's ~1250 checks.
    # With 128 vertices, the B&B tree could be huge. Let me think of a better approach.
    
    # Alternative: use a different upper bound.
    # At node with chosen set C and remaining set R:
    # UB = |C| + max independent set size in the induced subhypergraph on R
    # But that's as hard as the original problem.
    
    # Simpler UB: |C| + |R| (loose but fast)
    # Or: |C| + number of vertices in R that are individually compatible with C
    # The latter requires checking each vertex in R against C.
    
    # Let me try yet another approach: since N=128 is small, use a 
    # maximum clique formulation on the complement graph.
    # Actually, the problem is: find the largest subset with no forbidden triple.
    # This is the independence number of a 3-uniform hypergraph.
    # The complement problem: find the minimum vertex cover of the hypergraph.
    # By duality, alpha(H) = n - tau(H) where tau is the minimum vertex cover.
    # But computing tau for 3-uniform hypergraphs is also NP-hard.
    
    # Let me try a different tactic: use the fact that the forbidden condition
    # is quite structured. The condition gcd(a,b)==gcd(a,c)==gcd(b,c)==g means
    # that a=g*a', b=g*b', c=g*c' where gcd(a',b')=gcd(a',c')=gcd(b',c')=1... 
    # wait no, that's not right. gcd(a,b)=g means a=g*a', b=g*b' with gcd(a',b')=1.
    # Similarly gcd(a,c)=g means c=g*c' with gcd(a',c')=1, and gcd(b,c)=g means
    # gcd(b',c')=1.
    # So a forbidden triple with gcd g consists of g*a', g*b', g*c' where
    # gcd(a',b')=gcd(a',c')=gcd(b',c')=1.
    # 
    # This means: for a fixed g, the numbers that are multiples of g form a group,
    # and within that group, a forbidden triple is three numbers whose "reduced"
    # forms are pairwise coprime.
    #
    # So the problem decomposes by gcd value! For each g, consider the set of
    # numbers in [1,128] that are multiples of g. Within this set, we need to
    # avoid three numbers that are pairwise coprime when divided by g.
    #
    # Wait, but a number can be a multiple of multiple g's. The constraint is
    # global: no three numbers in our set can have all three pairwise gcds equal
    # to some g. So for each g, the three numbers must all be multiples of g,
    # and their quotients by g must be pairwise coprime.
    
    # This structure might allow a smarter approach, but for now let me just
    # do a well-optimized B&B.
    
    # Let me try a completely different approach: use a SAT-like or ILP-like
    # greedy with backtracking, but with strong pruning.
    
    # Actually, let me try the most straightforward thing: a well-implemented
    # branch and bound with the "compatible count" upper bound.
    
    # Key optimization: instead of checking all pairs in C for each candidate x,
    # maintain for each x a "conflict flag" that gets updated as we add to C.
    
    # Let me restructure:
    # - Process vertices in order of decreasing conflict_count
    # - Maintain chosen set as a list
    # - For each candidate x, check compatibility by iterating over pairs in chosen
    
    # To make this faster, let me precompute for each x the list of (a,b) codes
    # and use a set for chosen pairs.
    
    # Actually, let me just try the simple B&B and see if it's fast enough.
    
    def branch_and_bound(chosen, remaining_indices, depth=0):
        nonlocal best_solution, best_len
        
        if time.time() >= DEADLINE - 2:
            return
        
        # Upper bound
        ub = len(chosen) + len(remaining_indices)
        if ub <= best_len:
            return
        
        if not remaining_indices:
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_solution = chosen[:]
            return
        
        # Try to add as many as possible (greedy within this branch)
        # First, filter remaining to those compatible with chosen
        compatible = []
        chosen_set = set(chosen)
        for idx in remaining_indices:
            x = vertex_order[idx]
            if is_compatible(x, chosen_set):
                compatible.append(idx)
        
        # Tighter upper bound
        ub = len(chosen) + len(compatible)
        if ub <= best_len:
            return
        
        if not compatible:
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_solution = chosen[:]
            return
        
        # Choose the next vertex to branch on
        # Use the first compatible vertex (which has highest conflict count
        # since we ordered by decreasing conflict count)
        next_idx = compatible[0]
        x = vertex_order[next_idx]
        
        # Branch 1: include x
        new_chosen = chosen + [x]
        new_remaining = [i for i in compatible[1:] if is_compatible(vertex_order[i], set(new_chosen))]
        branch_and_bound(new_chosen, new_remaining, depth + 1)
        
        # Branch 2: exclude x
        branch_and_bound(chosen, compatible[1:], depth + 1)
    
    # Start B&B from the best greedy solution
    # Actually, let's start from scratch but use the greedy solution as initial best
    # and process in conflict order
    
    # The B&B as written explores include-first, which should find good solutions early
    # Let's run it with the full vertex order
    branch_and_bound([], list(range(len(vertex_order))))
    
    # Final validity check
    if not is_valid(best_solution):
        # Fallback
        selected = []
        S_set = set()
        for x in UNIVERSE:
            if can_add(selected, x):
                selected.append(x)
                S_set.add(x)
        if selected:
            best_solution = selected
        else:
            best_solution = [1]
    
    return best_solution


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
