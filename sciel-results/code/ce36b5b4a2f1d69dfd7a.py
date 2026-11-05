"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import deque
import time

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    Verify girth via BFS from identity.
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for 120s limit
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # For k=5, g=9: try random Cayley graphs on non-abelian groups
    # A Cayley graph on group G with generator set S (|S|=k, S=S^{-1}, e not in S)
    # is k-regular. It's bipartite if we can partition G into two sets.
    # For girth >= g, we need no non-trivial reduced word of length < g in S equals e.
    
    # Approach: Use G = S_m (symmetric group) for small m, or G = (Z/2Z)^n ⋊ H
    # Better: Use G = A_n or S_n for moderate n, with random involutions
    
    # For a bipartite Cayley graph, we need an element of order 2 that acts as
    # the "flip" between the two parts. Actually, for a general Cayley graph to be
    # bipartite, we need a homomorphism phi: G -> Z/2Z such that phi(s) = 1 for all s in S.
    
    # Let's try: G = S_n, generators are k random transpositions or products of disjoint transpositions
    # that all have the same parity (all even or all odd permutations)
    
    # Actually, for the Cayley graph to be bipartite with girth >= 9:
    # We need G with a subgroup H of index 2, and S ⊆ G\H with |S|=k, S=S^{-1}
    # (i.e., all generators are in G\H, so they're "odd" in some sense)
    
    # Let's try G = S_n with H = A_n (alternating group)
    # Generators must be odd permutations (transpositions or products of odd number of transpositions)
    # For girth 9, we need no word of length 1-8 in generators equals identity
    
    # Try increasing n until we find a valid set of generators
    
    def check_girth_cayley(group_elements, generators, g):
        """Check if Cayley graph has girth >= g using BFS from identity."""
        # group_elements: list of group elements, identity at index 0
        # generators: list of generator indices (into group_elements)
        # BFS from identity, track shortest path to each element
        # Girth >= g iff no element is reachable by two different paths of total length < g
        # More precisely: for each vertex, BFS gives distance. Girth >= g iff
        # for every edge (u,v), dist(u) + 1 + dist(v) >= g when considering
        # non-backtracking paths. Actually, simpler: girth >= g iff no cycle of length < g.
        # A cycle of length L corresponds to a word s1*s2*...*sL = e with si in S∪S^{-1}
        # and no consecutive si, si+1 are inverses (non-backtracking).
        
        # For a Cayley graph, we can do BFS from identity and check for early returns.
        # If we reach identity again at distance d < g, we have a cycle of length d.
        # But we need non-backtracking: the last step can't be inverse of first.
        
        # Simpler approach: BFS from identity, for each vertex at distance d, 
        # check if any neighbor is at distance d-1 (back edge) or < d-1 (cross edge creating shorter cycle).
        # Actually, the standard way: do BFS, and for each vertex v at distance d, 
        # if any neighbor u is at distance d' where d' <= d-2, then there's a cycle of length d+d'-1+1 = d+d' <= 2d-2.
        # Hmm, let me think more carefully.
        
        # For a Cayley graph with generators S, a cycle of length L exists iff there's a 
        # non-backtracking word s1...sL in S∪S^{-1} with product = e.
        
        # BFS approach: from identity, explore all non-backtracking walks up to length g-1.
        # If we return to identity at length < g, girth < g.
        
        # State: (current_group_element, previous_generator) where previous_generator is the
        # generator we used to get here (or None for start). Non-backtracking means we can't
        # use the inverse of previous_generator.
        
        n_elements = len(group_elements)
        # Map group elements to indices
        elem_to_idx = {elem: i for i, elem in enumerate(group_elements)}
        
        # Precompute: for each generator s, what is s^{-1}? In our case, generators are involutions,
        # so s^{-1} = s. So non-backtracking means we can't use the same generator twice in a row.
        
        # BFS: state = (elem_idx, last_gen_idx) where last_gen_idx is index in generators or -1
        # We want to know if we can reach (identity, any) at distance < g with last_gen != first_gen
        
        # Actually, let's just do BFS tracking (elem, last_gen) and see if we hit identity at dist < g
        # with last_gen != first_gen (for dist >= 2).
        
        # For dist 1: can't be identity (generators are not identity)
        # For dist 2: s1*s2 = e means s2 = s1^{-1} = s1 (since involutions), so s1=s2, which is backtracking. Not allowed.
        # For dist 3: s1*s2*s3 = e with s2 != s1, s3 != s2. Since involutions, this means s1*s2*s3 = e.
        
        # Let's do BFS up to depth g-1
        # State: (elem_idx, last_gen_idx)
        # Start: (0, -1) at distance 0
        
        visited = {}  # (elem_idx, last_gen_idx) -> distance
        queue = deque()
        queue.append((0, -1, 0))  # (elem_idx, last_gen_idx, dist)
        visited[(0, -1)] = 0
        
        while queue:
            elem_idx, last_gen, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for gen_idx, gen in enumerate(generators):
                if gen_idx == last_gen:
                    continue  # non-backtracking
                # Multiply elem * gen
                new_elem = group_elements[elem_idx] * gen
                new_idx = elem_to_idx.get(new_elem)
                if new_idx is None:
                    continue
                new_state = (new_idx, gen_idx)
                if new_state in visited:
                    continue
                visited[new_state] = dist + 1
                # Check if we've returned to identity
                if new_idx == 0:
                    # Cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                queue.append((new_idx, gen_idx, dist + 1))
        
        return True
    
    def build_symmetric_group(n):
        """Build S_n as a list of permutations (tuples)."""
        from itertools import permutations
        elems = list(permutations(range(n)))
        return elems
    
    def perm_mul(p, q):
        """Multiply permutations p and q (p after q, i.e., p∘q)."""
        return tuple(p[q[i]] for i in range(len(p)))
    
    # For efficiency, let's work with permutations as tuples and precompute multiplication
    
    # Try different group sizes
    best_result = None
    
    # For k=5, g=9, let's try S_n for n = 4, 5, 6, ...
    # We need 5 involutions that are all odd (to make bipartite) and satisfy girth constraint
    
    # In S_n, an involution is a product of disjoint transpositions.
    # It's odd if it's a product of an odd number of transpositions.
    # So for S_n, odd involutions are products of 1, 3, 5, ... transpositions.
    
    # For S_4: odd involutions are single transpositions (6 of them) and products of 3 transpositions (impossible in S_4 since max is 2 disjoint transpositions which is even).
    # Wait: in S_4, a transposition is odd (1 transposition). Product of 2 disjoint transpositions is even. So odd involutions in S_4 are just the 6 transpositions.
    # We need to choose 5 out of 6 transpositions.
    
    # Let's try S_4 first
    if k == 5 and g == 9:
        # Try S_4 with 5 transpositions
        from itertools import permutations
        S4 = list(permutations(range(4)))
        elem_to_idx = {p: i for i, p in enumerate(S4)}
        
        # All transpositions in S_4
        transpositions = []
        for i in range(4):
            for j in range(i+1, 4):
                p = list(range(4))
                p[i], p[j] = p[j], p[i]
                transpositions.append(tuple(p))
        
        # Try all combinations of 5 transpositions
        for combo in itertools.combinations(range(6), 5):
            gens = [transpositions[i] for i in combo]
            # Check girth
            # Precompute multiplication table for S4
            mul_table = [[None]*24 for _ in range(24)]
            for i, p in enumerate(S4):
                for j, q in enumerate(S4):
                    mul_table[i][j] = elem_to_idx[perm_mul(p, q)]
            
            # BFS for girth check
            # State: (elem_idx, last_gen_idx)
            visited = {}
            queue = deque()
            queue.append((0, -1, 0))
            visited[(0, -1)] = 0
            
            valid = True
            while queue and valid:
                elem_idx, last_gen, dist = queue.popleft()
                if dist >= g - 1:
                    continue
                for gen_idx, gen in enumerate(gens):
                    if gen_idx == last_gen:
                        continue
                    new_idx = mul_table[elem_idx][elem_to_idx[gen]]
                    new_state = (new_idx, gen_idx)
                    if new_state in visited:
                        continue
                    visited[new_state] = dist + 1
                    if new_idx == 0 and dist + 1 < g:
                        valid = False
                        break
                    queue.append((new_idx, gen_idx, dist + 1))
            
            if valid:
                # Build the Cayley graph
                # The graph is bipartite: even permutations vs odd permutations
                # Edges: for each vertex v and each generator s, edge (v, v*s)
                # To avoid duplicate edges, only add edge if v < v*s in some ordering
                # Or: since it's bipartite, add edges from even part to odd part only
                
                edges = []
                for i, v in enumerate(S4):
                    v_parity = 0  # will compute
                    # Compute parity of permutation
                    perm_list = list(v)
                    parity = 0
                    visited_pos = [False]*4
                    for idx in range(4):
                        if not visited_pos[idx]:
                            cycle_len = 0
                            cur = idx
                            while not visited_pos[cur]:
                                visited_pos[cur] = True
                                cur = perm_list[cur]
                                cycle_len += 1
                            parity += (cycle_len - 1)
                    v_is_odd = (parity % 2 == 1)
                    
                    if v_is_odd:
                        continue  # Only add edges from even to odd
                    
                    for gen in gens:
                        w = perm_mul(v, gen)
                        w_idx = elem_to_idx[w]
                        if i < w_idx:
                            edges.append((i, w_idx))
                        else:
                            edges.append((w_idx, i))
                
                # Deduplicate
                edge_set = set()
                unique_edges = []
                for e in edges:
                    if e not in edge_set:
                        edge_set.add(e)
                        unique_edges.append(e)
                
                return 24, unique_edges
        
        # If S_4 didn't work, try S_5
        # In S_5, odd involutions: single transpositions (10) and products of 3 disjoint transpositions (impossible in S_5, max 2 disjoint)
        # Wait, in S_5 we can have at most floor(5/2)=2 disjoint transpositions, so only single transpositions are odd involutions.
        # We need 5 out of 10 transpositions.
        
        from itertools import permutations
        S5 = list(permutations(range(5)))
        elem_to_idx = {p: i for i, p in enumerate(S5)}
        
        transpositions = []
        for i in range(5):
            for j in range(i+1, 5):
                p = list(range(5))
                p[i], p[j] = p[j], p[i]
                transpositions.append(tuple(p))
        
        # 10 transpositions, choose 5: C(10,5) = 252 combinations
        # Precompute multiplication table
        mul_table = [[None]*120 for _ in range(120)]
        for i, p in enumerate(S5):
            for j, q in enumerate(S5):
                mul_table[i][j] = elem_to_idx[perm_mul(p, q)]
        
        # Try random subsets first, then all if time permits
        random.seed(42)
        tried = set()
        
        for attempt in range(200):
            if time.time() - start_time > time_limit - 10:
                break
            combo = tuple(sorted(random.sample(range(10), 5)))
            if combo in tried:
                continue
            tried.add(combo)
            
            gens = [transpositions[i] for i in combo]
            gen_indices = [elem_to_idx[g] for g in gens]
            
            # BFS for girth check
            visited = {}
            queue = deque()
            queue.append((0, -1, 0))
            visited[(0, -1)] = 0
            
            valid = True
            while queue and valid:
                elem_idx, last_gen, dist = queue.popleft()
                if dist >= g - 1:
                    continue
                for gen_idx, gen_idx_val in enumerate(gen_indices):
                    if gen_idx == last_gen:
                        continue
                    new_idx = mul_table[elem_idx][gen_idx_val]
                    new_state = (new_idx, gen_idx)
                    if new_state in visited:
                        continue
                    visited[new_state] = dist + 1
                    if new_idx == 0 and dist + 1 < g:
                        valid = False
                        break
                    queue.append((new_idx, gen_idx, dist + 1))
            
            if valid:
                # Build the Cayley graph
                edges = []
                for i, v in enumerate(S5):
                    # Compute parity
                    perm_list = list(v)
                    parity = 0
                    visited_pos = [False]*5
                    for idx in range(5):
                        if not visited_pos[idx]:
                            cycle_len = 0
                            cur = idx
                            while not visited_pos[cur]:
                                visited_pos[cur] = True
                                cur = perm_list[cur]
                                cycle_len += 1
                            parity += (cycle_len - 1)
                    v_is_odd = (parity % 2 == 1)
                    
                    if v_is_odd:
                        continue
                    
                    for gen_idx_val in gen_indices:
                        w_idx = mul_table[i][gen_idx_val]
                        if i < w_idx:
                            edges.append((i, w_idx))
                        else:
                            edges.append((w_idx, i))
                
                edge_set = set()
                unique_edges = []
                for e in edges:
                    if e not in edge_set:
                        edge_set.add(e)
                        unique_edges.append(e)
                
                return 120, unique_edges
        
        # Try S_6
        if time.time() - start_time < time_limit - 20:
            from itertools import permutations
            S6 = list(permutations(range(6)))
            elem_to_idx = {p: i for i, p in enumerate(S6)}
            
            # Odd involutions in S_6: single transpositions (15) and products of 3 disjoint transpositions
            transpositions = []
            for i in range(6):
                for j in range(i+1, 6):
                    p = list(range(6))
                    p[i], p[j] = p[j], p[i]
                    transpositions.append(tuple(p))
            
            # Products of 3 disjoint transpositions
            triple_transpositions = []
            for a in range(6):
                for b in range(a+1, 6):
                    for c in range(b+1, 6):
                        d = -1
                        for e in range(6):
                            if e not in [a,b,c]:
                                d = e
                                break
                        if d == -1:
                            continue
                        # Need one more: f
                        for e in range(6):
                            if e not in [a,b,c,d]:
                                f = e
                                break
                        if f == -1:
                            continue
                        p = list(range(6))
                        p[a], p[b] = p[b], p[a]
                        p[c], p[d] = p[d], p[c]
                        p[e], p[f] = p[f], p[e]
                        triple_transpositions.append(tuple(p))
            
            all_odd_involutions = transpositions + triple_transpositions
            
            # Precompute multiplication table (720x720 = 518400 entries, should be OK)
            mul_table = [[None]*720 for _ in range(720)]
            for i, p in enumerate(S6):
                for j, q in enumerate(S6):
                    mul_table[i][j] = elem_to_idx[perm_mul(p, q)]
            
            # Try random subsets of 5 from all_odd_involutions
            random.seed(123)
            tried = set()
            
            for attempt in range(100):
                if time.time() - start_time > time_limit - 5:
                    break
                combo = tuple(sorted(random.sample(range(len(all_odd_involutions)), 5)))
                if combo in tried:
                    continue
                tried.add(combo)
                
                gens = [all_odd_involutions[i] for i in combo]
                gen_indices = [elem_to_idx[g] for g in gens]
                
                visited = {}
                queue = deque()
                queue.append((0, -1, 0))
                visited[(0, -1)] = 0
                
                valid = True
                while queue and valid:
                    elem_idx, last_gen, dist = queue.popleft()
                    if dist >= g - 1:
                        continue
                    for gen_idx, gen_idx_val in enumerate(gen_indices):
                        if gen_idx == last_gen:
                            continue
                        new_idx = mul_table[elem_idx][gen_idx_val]
                        new_state = (new_idx, gen_idx)
                        if new_state in visited:
                            continue
                        visited[new_state] = dist + 1
                        if new_idx == 0 and dist + 1 < g:
                            valid = False
                            break
                        queue.append((new_idx, gen_idx, dist + 1))
                
                if valid:
                    edges = []
                    for i, v in enumerate(S6):
                        perm_list = list(v)
                        parity = 0
                        visited_pos = [False]*6
                        for idx in range(6):
                            if not visited_pos[idx]:
                                cycle_len = 0
                                cur = idx
                                while not visited_pos[cur]:
                                    visited_pos[cur] = True
                                    cur = perm_list[cur]
                                    cycle_len += 1
                                parity += (cycle_len - 1)
                        v_is_odd = (parity % 2 == 1)
                        
                        if v_is_odd:
                            continue
                        
                        for gen_idx_val in gen_indices:
                            w_idx = mul_table[i][gen_idx_val]
                            if i < w_idx:
                                edges.append((i, w_idx))
                            else:
                                edges.append((w_idx, i))
                    
                    edge_set = set()
                    unique_edges = []
                    for e in edges:
                        if e not in edge_set:
                            edge_set.add(e)
                            unique_edges.append(e)
                    
                    return 720, unique_edges
    
    # Fallback: L-U-W construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
