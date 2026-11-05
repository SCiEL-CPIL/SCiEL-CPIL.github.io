import time
from math import gcd

def construct():
    deadline = time.time() + 85  # 85s to leave margin for safety
    
    n_max = 128
    vertices = list(range(1, n_max + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i + 1, n_max + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    # Find all bad triples (hyperedges)
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = set()
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            for c in range(b + 1, n_max + 1):
                g1 = gcd_table[a][b]
                g2 = gcd_table[a][c]
                g3 = gcd_table[b][c]
                if g1 == g2 == g3:
                    bad_triples.add((a, b, c))
    
    # Build conflict graph: two vertices conflict if they appear together in a bad triple
    # Actually for independent set in hypergraph, we need: a set is independent if no hyperedge is fully contained
    # For branch and bound, we work with the hypergraph directly
    
    # For each vertex, find which other pairs would form a bad triple with it
    # vertex_conflicts[v] = set of (u, w) such that (v, u, w) is a bad triple
    vertex_pairs = {}
    for v in vertices:
        vertex_pairs[v] = set()
    
    for (a, b, c) in bad_triples:
        vertex_pairs[a].add((b, c))
        vertex_pairs[b].add((a, c))
        vertex_pairs[c].add((a, b))
    
    # For branch and bound, we need to track which vertices are "blocked"
    # If we include vertex v, then any pair (u,w) in vertex_pairs[v] cannot both be included
    # This creates a conflict graph where u and w conflict (given v is included)
    
    # Better approach: Use the standard MIS formulation
    # Build the "conflict graph" where an edge exists between u and w if there exists v 
    # such that (v,u,w) is a bad triple. But this isn't quite right either.
    
    # Let me use a different formulation:
    # A set S is valid if for every bad triple (a,b,c), not all three are in S.
    # This is the complement of: for every bad triple, at least one is excluded.
    # So we want to maximize |S| such that S contains no complete hyperedge.
    
    # For branch and bound on hypergraph independent set:
    # State: (remaining_vertices, blocked_pairs)
    # blocked_pairs: set of pairs (u,w) that cannot both be in the solution
    # (because some vertex v that's already included would form a bad triple with u,w)
    
    # Actually, let me use a simpler but effective approach:
    # Greedy coloring bound + branch and bound
    
    # First, build a conflict graph for upper bounding
    # Two vertices u, w are "in conflict" if there exists some v such that (v,u,w) is a bad triple
    # This gives an upper bound via graph coloring
    
    conflict_graph = [[False] * (n_max + 1) for _ in range(n_max + 1)]
    for (a, b, c) in bad_triples:
        conflict_graph[a][b] = True
        conflict_graph[b][a] = True
        conflict_graph[a][c] = True
        conflict_graph[c][a] = True
        conflict_graph[b][c] = True
        conflict_graph[c][b] = True
    
    # Degree of each vertex in conflict graph
    degree = [0] * (n_max + 1)
    for v in vertices:
        for u in vertices:
            if u != v and conflict_graph[v][u]:
                degree[v] += 1
    
    # Greedy coloring for upper bound
    def greedy_color_bound(verts):
        """Return upper bound on independent set size using greedy coloring."""
        if not verts:
            return 0
        # Greedy coloring: assign colors greedily
        color = {}
        colors_used = 0
        for v in verts:
            # Find smallest available color
            used_colors = set()
            for u in verts:
                if u != v and conflict_graph[v][u] and u in color:
                    used_colors.add(color[u])
            c = 1
            while c in used_colors:
                c += 1
            color[v] = c
            colors_used = max(colors_used, c)
        return colors_used
    
    # Branch and bound
    best_solution = []
    best_size = 0
    
    # Initial greedy solution for lower bound
    def greedy_solution(order):
        selected = []
        # Track which pairs are blocked
        blocked = set()
        for v in order:
            # Check if adding v creates a bad triple
            valid = True
            for u in selected:
                for w in selected:
                    if u == w:
                        continue
                    # Check if (v, u, w) is a bad triple
                    if v < u:
                        t = (v, u, w) if v < u < w else (v, w, u) if v < w < u else None
                        if t and t in bad_triples:
                            valid = False
                            break
                    elif u < v:
                        t = (u, v, w) if u < v < w else (u, w, v) if u < w < v else None
                        if t and t in bad_triples:
                            valid = False
                            break
                    else:
                        t = (u, v, w) if u < v < w else (u, w, v) if u < w < v else None
                        if t and t in bad_triples:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                selected.append(v)
        return selected
    
    # Better: use the vertex_pairs structure
    def can_add(v, selected):
        """Check if v can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                if (v, a, b) in bad_triples or (a, v, b) in bad_triples or (a, b, v) in bad_triples:
                    return False
        return True
    
    # Try several greedy orderings to get a good initial lower bound
    strategies = []
    strategies.append(list(range(1, n_max + 1)))
    strategies.append(list(range(n_max, 0, -1)))
    strategies.append(sorted(vertices, key=lambda x: degree[x]))
    strategies.append(sorted(vertices, key=lambda x: degree[x], reverse=True))
    
    for order in strategies:
        sol = greedy_solution(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]
    
    # Now do branch and bound
    # We'll use a recursive approach with memoization on the remaining set (as a frozenset)
    # But 128 vertices is too many for full memoization. Use partial memoization.
    
    # Use a different approach: branch on vertices in order of degree
    # With strong pruning
    
    def branch_and_bound(remaining, current_set, current_size):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        if not remaining:
            if current_size > best_size:
                best_size = current_size
                best_solution = current_set[:]
            return
        
        # Upper bound: current_size + greedy_color_bound(remaining)
        if current_size + greedy_color_bound(remaining) <= best_size:
            return
        
        # Also: if current_size + len(remaining) <= best_size, prune
        if current_size + len(remaining) <= best_size:
            return
        
        # Choose vertex to branch on (highest degree in remaining)
        v = max(remaining, key=lambda x: degree[x])
        
        # Branch 1: include v
        new_remaining = [x for x in remaining if x != v]
        # Check if v is compatible with current_set
        if can_add(v, current_set):
            current_set.append(v)
            branch_and_bound(new_remaining, current_set, current_size + 1)
            current_set.pop()
        
        # Branch 2: exclude v
        branch_and_bound(new_remaining, current_set, current_size)
    
    # The simple B&B above might be too slow. Let me use a more sophisticated approach.
    # Use the fact that the problem has structure.
    
    # Alternative: Use maximum independent set on the conflict graph as an upper bound,
    # and use a specialized search.
    
    # Let me try a different structural approach:
    # The bad triples have structure. Let's analyze what makes a triple bad.
    # gcd(a,b) = gcd(a,c) = gcd(b,c) = g
    # This means g divides a, b, c, and after dividing by g, the three numbers are pairwise coprime... no.
    # gcd(a/g, b/g) = 1, gcd(a/g, c/g) = 1, gcd(b/g, c/g) = 1
    # So a/g, b/g, c/g are pairwise coprime.
    
    # This means: for a fixed g, the numbers divisible by g form a set where we can't have
    # three numbers that are pairwise coprime after dividing by g.
    
    # This is a much more structured problem! Let's exploit this.
    
    # For each g from 1 to 128:
    #   Let S_g = {x in [1,128] : g | x}
    #   For x in S_g, let x' = x/g
    #   We cannot have three elements x1, x2, x3 in S_g such that x1/g, x2/g, x3/g are pairwise coprime.
    
    # So the constraint is: for each g, the set {x/g : x in selected, g|x} must not contain
    # three pairwise coprime numbers.
    
    # A set of numbers contains no three pairwise coprime elements iff... 
    # This is related to the Erdos-Selfridge type results.
    
    # Key insight: If we look at the "reduced" values x/g, we need them to be "2-wise intersecting" in some sense.
    # Specifically, no three should be pairwise coprime.
    
    # A sufficient condition: all reduced values share a common prime factor, or more generally,
    # the set can be covered by 2 sets each with a common prime factor.
    
    # Actually, let's think about it differently. For a fixed g, define T_g = {x/g : x in selected, g|x}.
    # We need T_g to have no three pairwise coprime elements.
    
    # This is equivalent to: the maximum size of a subset of T_g with pairwise coprime elements is at most 2.
    
    # Let me precompute for each g what the constraint is.
    
    # New approach: Use the structural insight to build a smarter greedy + local search.
    
    # For each number x in [1,128], and for each g that divides x, x contributes x/g to the "reduced set" for g.
    # The constraint is that for each g, the reduced set has no 3 pairwise coprime elements.
    
    # Let's define: for a candidate set S, for each g, R_g(S) = {x/g : x in S, g|x}.
    # Constraint: |{pairwise coprime subset of R_g(S)}| <= 2 for all g.
    
    # A set A has no 3 pairwise coprime elements iff for every triple (a,b,c) in A, 
    # at least one pair has gcd > 1.
    
    # This is the same as our original constraint! So we haven't simplified, but the structure
    # (grouping by g) might help with search.
    
    # Let me try a completely different approach: since the problem has algebraic structure,
    # let's think about what sets are valid.
    
    # Claim: The set of all numbers with a fixed prime factor p is valid.
    # Proof: If a, b, c all divisible by p, then gcd(a,b) >= p, gcd(a,c) >= p, gcd(b,c) >= p.
    # For them to be a bad triple, we need gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    # Then g >= p. Let a = g*a', b = g*b', c = g*c'. Then gcd(a',b') = gcd(a',c') = gcd(b',c') = 1.
    # And p | g, so p | a', p | b', p | c'. But then gcd(a',b') >= p > 1, contradiction.
    # Wait, that's not right. p | g means p divides g, and a = g*a', so p | a. But a' = a/g, so p may not divide a'.
    
    # Let me redo: a = g*a', and p | a, p | g. So a' = a/g. Since p | a and p | g, p may or may not divide a'.
    # Example: a = 12, g = 6, p = 2. a' = 2, p | a'. 
    # Example: a = 12, g = 6, p = 3. a' = 2, p does not divide a'.
    
    # So the claim is false in general. Let me check: {6, 10, 15}. gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. Not all equal.
    # {6, 12, 18}: gcd(6,12)=6, gcd(6,18)=6, gcd(12,18)=6. All equal to 6! So {6,12,18} is a bad triple.
    # They all have prime factor 2 and 3. So the claim "all numbers with fixed prime factor p is valid" is FALSE.
    
    # OK so let me go back to the branch and bound but make it more efficient.
    
    # Key optimization: Instead of checking all pairs in selected, maintain a "conflict set" 
    # that tracks which pairs are forbidden.
    
    # Let me implement a cleaner branch and bound with the following:
    # 1. Order vertices by some heuristic
    # 2. Maintain a "forbidden pairs" set
    # 3. Use upper bounds based on the conflict graph
    
    # Actually, let me try yet another approach: since we have 128 vertices and the time limit is 85s,
    # let's use a randomized local search with many restarts, but with a smarter move set.
    
    # The current program scores 1.75. The inspirations score 1.875. Let me try to understand
    # what they do differently.
    
    # Let me implement a robust randomized local search with:
    # 1. Multiple diverse initial solutions
    # 2. 1-for-1, 1-for-2, 2-for-1, 2-for-2 exchanges
    # 3. Simulated annealing temperature schedule
    # 4. Many restarts
    
    import random
    
    def get_conflict_pairs(v, selected):
        """Get pairs in selected that would form a bad triple with v."""
        conflicts = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                if (v, a, b) in bad_triples or (a, v, b) in bad_triples or (a, b, v) in bad_triples:
                    conflicts.append((a, b))
        return conflicts
    
    def is_valid_set(s):
        """Check if a set is valid (no bad triple)."""
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                for k in range(j + 1, len(s)):
                    a, b, c = s[i], s[j], s[k]
                    if (a, b, c) in bad_triples:
                        return False
        return True
    
    # Fast validity check using precomputed data
    def count_bad_triples_in_set(s):
        """Count bad triples in set s."""
        count = 0
        s_set = set(s)
        for (a, b, c) in bad_triples:
            if a in s_set and b in s_set and c in s_set:
                count += 1
        return count
    
    # For local search, we need efficient add/remove operations
    # Precompute for each vertex v, the list of bad triples containing v
    triples_with_vertex = {v: [] for v in vertices}
    for (a, b, c) in bad_triples:
        triples_with_vertex[a].append((b, c))
        triples_with_vertex[b].append((a, c))
        triples_with_vertex[c].append((a, b))
    
    # State: selected set, and for each pair (u,w) in selected, count how many v would make (v,u,w) a bad triple
    # This is expensive to maintain. Let's use a simpler approach.
    
    # Efficient local search:
    # Maintain: selected (list), in_selected (boolean array)
    # For adding x: check all pairs in selected for bad triple with x
    # For removing x: just remove
    # For swap: remove x, add y
    
    def local_search(selected, time_limit, rng):
        """Local search with 1-for-1, 1-for-2, 2-for-1, 2-for-2 moves."""
        selected = list(selected)
        in_sel = [False] * (n_max + 1)
        for v in selected:
            in_sel[v] = True
        remaining = [v for v in vertices if not in_sel[v]]
        
        start_time = time.time()
        current_size = len(selected)
        best_local = selected[:]
        best_local_size = current_size
        
        # Precompute for faster can_add: for each v, list of (u,w) pairs that form bad triple with v
        # Already have triples_with_vertex
        
        def try_add(x, sel, in_s):
            """Try to add x to sel. Returns True if successful."""
            for (u, w) in triples_with_vertex[x]:
                if in_s[u] and in_s[w]:
                    return False
            sel.append(x)
            in_s[x] = True
            return True
        
        def try_remove(x, sel, in_s):
            """Remove x from sel."""
            sel.remove(x)
            in_s[x] = False
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try 1-for-0: add a single element
            for x in remaining:
                if try_add(x, selected, in_sel):
                    remaining.remove(x)
                    improved = True
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = selected[:]
                    break
            
            if not improved:
                # Try 1-for-1: remove one, add one
                found = False
                for i, y in enumerate(selected):
                    try_remove(y, selected, in_sel)
                    for x in remaining:
                        if try_add(x, selected, in_sel):
                            remaining.remove(x)
                            remaining.append(y)
                            found = True
                            improved = True
                            break
                    if not found:
                        try_remove(x if False else y, selected, in_sel) if False else None
                        # Put y back
                        selected.append(y)
                        in_sel[y] = True
                    else:
                        break
                if not found:
                    # Try 1-for-2: remove one, add two
                    found = False
                    for i, y in enumerate(selected):
                        try_remove(y, selected, in_sel)
                        for x1 in remaining:
                            if try_add(x1, selected, in_sel):
                                remaining_copy = remaining[:]
                                remaining_copy.remove(x1)
                                for x2 in remaining_copy:
                                    if try_add(x2, selected, in_sel):
                                        remaining.remove(x1)
                                        rem_idx = remaining.index(x2) if x2 in remaining else -1
                                        if rem_idx >= 0:
                                            remaining.pop(rem_idx)
                                        remaining.append(y)
                                        found = True
                                        improved = True
                                        current_size += 1
                                        if current_size > best_local_size:
                                            best_local_size = current_size
                                            best_local = selected[:]
                                        break
                                if not found:
                                    # Undo x1
                                    selected.remove(x1)
                                    in_sel[x1] = True
                            if found:
                                break
                        if not found:
                            # Put y back
                            selected.append(y)
                            in_sel[y] = True
                        else:
                            break
                    if not found:
                        # Try 2-for-1: remove two, add one
                        found = False
                        n_sel = len(selected)
                        for i in range(n_sel):
                            for j in range(i + 1, n_sel):
                                y1, y2 = selected[i], selected[j]
                                # Remove y1 and y2
                                selected.remove(y1)
                                selected.remove(y2)
                                in_sel[y1] = False
                                in_sel[y2] = False
                                for x in remaining:
                                    if try_add(x, selected, in_sel):
                                        remaining.remove(x)
                                        remaining.append(y1)
                                        remaining.append(y2)
                                        found = True
                                        improved = True
                                        current_size -= 1
                                        break
                                if not found:
                                    # Put y1, y2 back
                                    selected.append(y1)
                                    selected.append(y2)
                                    in_sel[y1] = True
                                    in_sel[y2] = True
                                else:
                                    break
                            if found:
                                break
                        if not found:
                            # Try 2-for-2: remove two, add two
                            found = False
                            n_sel = len(selected)
                            for i in range(n_sel):
                                for j in range(i + 1, n_sel):
                                    y1, y2 = selected[i], selected[j]
                                    selected.remove(y1)
                                    selected.remove(y2)
                                    in_sel[y1] = False
                                    in_sel[y2] = False
                                    for x1 in remaining:
                                        if try_add(x1, selected, in_sel):
                                            rem_list = remaining[:]
                                            rem_list.remove(x1)
                                            for x2 in rem_list:
                                                if try_add(x2, selected, in_sel):
                                                    remaining.remove(x1)
                                                    idx = remaining.index(x2) if x2 in remaining else -1
                                                    if idx >= 0:
                                                        remaining.pop(idx)
                                                    remaining.append(y1)
                                                    remaining.append(y2)
                                                    found = True
                                                    improved = True
                                                    break
                                        if not found:
                                            selected.remove(x1)
                                            in_sel[x1] = True
                                        else:
                                            break
                                    if not found:
                                        selected.append(y1)
                                        selected.append(y2)
                                        in_sel[y1] = True
                                        in_sel[y2] = True
                                    else:
                                        break
                                if found:
                                    break
                            if not found:
                                break
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                break
        
        return best_local
    
    # Run multiple randomized greedy + local search
    rng = random.Random(42)
    best_overall = best_solution[:]
    best_overall_size = best_size
    
    # Generate diverse initial solutions
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy
        order = vertices[:]
        rng.shuffle(order)
        # Bias towards low-degree vertices
        order.sort(key=lambda x: (degree[x], rng.random()))
        
        selected = []
        in_sel = [False] * (n_max + 1)
        for v in order:
            # Check if v can be added
            can = True
            for (u, w) in triples_with_vertex[v]:
                if in_sel[u] and in_sel[w]:
                    can = False
                    break
            if can:
                selected.append(v)
                in_sel[v] = True
        
        # Local search
        if time.time() > deadline - 5:
            break
        
        time_left = deadline - time.time() - 3
        if time_left < 1:
            break
        
        result = local_search(selected, min(time_left, 2.0), rng)
        if len(result) > best_overall_size:
            best_overall_size = len(result)
            best_overall = result[:]
    
    # Final local search on the best solution found
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 3
        if time_left > 1:
            result = local_search(best_overall, time_left, rng)
            if len(result) > best_overall_size:
                best_overall = result
    
    return best_overall
