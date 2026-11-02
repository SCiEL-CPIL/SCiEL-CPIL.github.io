import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    if k == 5 and g <= 12:
        try:
            q = 13
            # Build PSL(2,13) elements: matrices [a b; c d] with det = 1, mod q
            psl_elements = []
            seen = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            mat = (a, b, c, d)
                            neg = (-a % q, -b % q, -c % q, -d % q)
                            canon = mat if mat < neg else neg
                            if canon not in seen:
                                seen.add(canon)
                                psl_elements.append(canon)
                        else:
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                mat = (a, b, c, d)
                                neg = (-a % q, -b % q, -c % q, -d % q)
                                canon = mat if mat < neg else neg
                                if canon not in seen:
                                    seen.add(canon)
                                    psl_elements.append(canon)

            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)

            def mat_mult(u, s):
                ua, ub, uc, ud = u
                sa, sb, sc, sd = s
                return ((ua * sa + ub * sc) % q,
                        (ua * sb + ub * sd) % q,
                        (uc * sa + ud * sc) % q,
                        (uc * sb + ud * sd) % q)

            # Find involutions: elements of order 2
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    m2 = mat_mult(e, e)
                    if m2 == identity:
                        involutions.append(e)

            n_inv = len(involutions)

            # Precompute prod_table[u_idx][j] = index of (psl_elements[u_idx] * involutions[j])
            prod_table = np.empty((n_vertices, n_inv), dtype=np.int32)
            for u_idx, u in enumerate(psl_elements):
                for j, s in enumerate(involutions):
                    prod_table[u_idx, j] = idx_map[mat_mult(u, s)]

            # Precompute pair products for 4-cycle check
            pair_is_inv = np.zeros((n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(i + 1, n_inv):
                    prod = mat_mult(involutions[i], involutions[j])
                    prod2 = mat_mult(prod, prod)
                    if prod2 == identity:
                        pair_is_inv[i, j] = True
                        pair_is_inv[j, i] = True

            # Precompute triple products for 6-cycle check
            triple_is_inv = np.zeros((n_inv, n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(n_inv):
                    if i == j:
                        continue
                    ij = mat_mult(involutions[i], involutions[j])
                    for l in range(n_inv):
                        if l == i or l == j:
                            continue
                        ijl = mat_mult(ij, involutions[l])
                        ijl2 = mat_mult(ijl, ijl)
                        if ijl2 == identity:
                            triple_is_inv[i, j, l] = True

            # Fast girth check using non-backtracking walk enumeration
            # For a k-regular graph with girth >= g (odd, g=2r+1), 
            # we need to verify no non-backtracking walk of length < g-1 returns to start
            # For g=9, r=4, we check walks of length up to 7 (but really up to 2r-1=7)
            # Actually for odd girth g=2r+1, the girth condition is equivalent to:
            # no non-backtracking closed walk of length < 2r (i.e., < g-1)
            # Wait, let me reconsider. For girth >= g (odd), we need:
            # The number of non-backtracking walks of length l from any vertex
            # equals k*(k-1)^(l-1) for l < g.
            # A simpler check: no non-backtracking walk of length < g-1 is closed.
            # For g=9, we check lengths 1 through 7 (but length 1 is trivially not closed in simple graph)
            # Length 2: u->v->u is non-backtracking? No, it IS backtracking.
            # Non-backtracking means you can't immediately go back.
            # So a non-backtracking walk of length 2 is u->v->w where w != u.
            # A non-backtracking closed walk of length 2 would be u->v->u, but that's backtracking.
            # So the shortest non-backtracking closed walks have length >= 3 (triangles).
            # For girth >= 9, we need no non-backtracking closed walk of length 3,4,5,6,7,8.
            # But even-length non-backtracking closed walks correspond to cycles in the bipartite double cover.
            # 
            # Simpler approach: BFS from each vertex up to depth (g-1)//2 = 4
            # If any vertex is reached by two different paths of total length < g, we have a short cycle.
            #
            # Let me use the standard approach: for each start vertex, do a BFS tracking
            # the first parent. If we encounter a vertex that was already visited at a shallower
            # depth via a different path, the cycle length is dist[u] + dist[v] + 1.
            # But this isn't quite right for non-backtracking.
            #
            # The cleanest method for this specific structure:
            # Since we have prod_table, we can enumerate non-backtracking walks directly.
            # A non-backtracking walk is (v0, v1, ..., vl) where vi+1 != vi-1 for all i.
            # For our Cayley graph with generators S (involutions), a walk is determined by
            # the sequence of generators. Since generators are involutions, the walk
            # v0, v0*s1, v0*s1*s2, ... is non-backtracking iff s_i != s_{i-1} for all i.
            # A non-backtracking closed walk of length l means s1*s2*...*sl = identity
            # with s_i != s_{i-1}.
            #
            # For girth >= 9, we need no non-backtracking closed walk of length < 9.
            # This means: no product s1*s2*...*sl = identity with s_i != s_{i-1} for l = 3,4,5,6,7,8.
            #
            # We can precompute for our 5 generators whether any such short product equals identity.
            # But this depends on the specific set S chosen.
            #
            # Alternative: just do a bounded BFS from a few representative vertices.
            # Since the graph is vertex-transitive (Cayley graph), checking from vertex 0
            # is sufficient!
            #
            # For a vertex-transitive graph, girth >= g iff from vertex 0, no non-backtracking
            # closed walk of length < g exists.
            #
            # Let's enumerate all non-backtracking walks from vertex 0 up to length 8.
            # A non-backtracking walk of length l is a sequence of l generators (s1,...,sl)
            # with s_i != s_{i-1}. The endpoint is 0 * s1 * s2 * ... * sl.
            # It's closed if the endpoint is 0.

            def check_girth_nb(S_indices):
                """Check girth >= g using non-backtracking walk enumeration from vertex 0.
                Since the graph is vertex-transitive, this suffices."""
                S = list(S_indices)
                # Precompute prefix products for speed
                # We'll enumerate all non-backtracking sequences of length 3 to g-1
                # and check if any product equals identity
                
                # For length 3: s1*s2*s3 = id, s1!=s2, s2!=s3
                # This is equivalent to s3 = s2*s1 (since involutions), and s3 != s2, s1 != s2
                # i.e., s2*s1 != s2 (always true since s1!=id) and s1 != s2
                # Wait, s3 = s1*s2 (involutions: s3 = (s1*s2)^{-1} = s2*s1)
                # So we need s1*s2 = s3, with s1!=s2, s2!=s3
                # s2 != s3 means s2 != s1*s2, which means s1 != id (true)
                # So any pair s1 != s2 gives a potential 3-cycle if s1*s2 is also in S
                # and s1*s2 != s1 and s1*s2 != s2
                # s1*s2 != s1 iff s2 != id (true)
                # s1*s2 != s2 iff s1 != id (true)
                # So: 3-cycle exists iff there exist s1,s2 in S with s1!=s2 and s1*s2 in S
                
                # Let's just do a direct enumeration for lengths 3 through g-1
                # Number of non-backtracking walks of length l from a fixed start:
                # k * (k-1)^(l-1) = 5 * 4^(l-1)
                # For l=8: 5*4^7 = 5*16384 = 81920
                # Total for l=3..8: 5*(4^2+4^3+4^4+4^5+4^6+4^7) = 5*(16+64+256+1024+4096+16384) = 5*21840 = 109200
                # Each check is O(l) to compute the product, but we can do it incrementally.
                
                # Use iterative DFS to enumerate non-backtracking walks
                # State: (current_vertex, last_generator_index, length)
                # We track the product so far to check if it equals identity at the end
                
                # Actually, let's just track the current group element.
                # Start at identity (vertex 0).
                # At each step, multiply by a generator different from the last one used.
                # Check if we return to identity at length < g.
                
                # For efficiency, we only need to check lengths 3 through g-1.
                # Length 1: can't be closed (no generator is identity)
                # Length 2: s1*s1 = id, but s2 must != s1, so s1*s2 != id unless s1=s2 which is forbidden
                #   Wait, s1*s2 = id iff s2 = s1^{-1} = s1 (involution), but s2 != s1 is required.
                #   So length 2 non-backtracking walks are never closed. Good.
                # Length 3: s1*s2*s3 = id with s1!=s2, s2!=s3
                #   s3 = s2*s1 (since s3 = (s1*s2)^{-1} = s2*s1)
                #   Need s3 in S, s3 != s2, s1 != s2
                #   s3 != s2 iff s2*s1 != s2 iff s1 != id (true)
                #   So: exists s1,s2 in S, s1!=s2, s1*s2 in S, s1*s2 != s1 (iff s2!=id, true), s1*s2 != s2 (iff s1!=id, true)
                #   Simplifies to: exists s1,s2 in S with s1!=s2 and s1*s2 in S
                
                # Let me just do the enumeration directly. It's at most ~110K steps which is fast.
                
                # Build a lookup: for each pair of distinct generators, what's their product?
                # prod_S[i][j] = index in S of (involutions[S[i]] * involutions[S[j]]) or -1 if not in S
                S_set = set(S)
                S_to_idx = {s: i for i, s in enumerate(S)}
                
                # prod_in_S[i][j] = index in S of S[i]*S[j] if in S, else -1
                prod_in_S = np.full((5, 5), -1, dtype=np.int32)
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            continue
                        prod = mat_mult(involutions[S[i]], involutions[S[j]])
                        if prod in idx_map:
                            prod_idx = idx_map[prod]
                            # Check if prod is in S
                            for si in S:
                                if involutions[si] == prod:
                                    prod_in_S[i, j] = si
                                    break
                
                # Now enumerate non-backtracking walks using the reduced state space
                # State: (current_element_index_in_psl, last_gen_index)
                # But that's n_vertices * 5 states, too many.
                # 
                # Better: just track the product so far as a group element.
                # Use a dict or array indexed by psl element.
                
                # Let's do a BFS-like enumeration level by level.
                # Level l: set of (product, last_gen) pairs
                # Start: level 0: {(identity, -1)}
                # Level 1: {(s, s) for s in S}
                # Level l+1: for each (p, last) in level l, for each s in S with s != last:
                #   new_p = p * s, add (new_p, s) to level l+1
                # Check at each level l >= 3: is (identity, s) in level l for any s?
                
                # To keep memory bounded, we can use a set of (product_idx, last_gen) tuples.
                # But the number grows as 5*4^(l-1). For l=8, that's 81920 entries.
                # Each entry is a tuple of two ints. Should be fine.
                
                # Actually, we can be smarter. We only care about whether identity appears.
                # So at each level, we just need to track which (product, last_gen) pairs exist.
                
                # Let me use a different approach: since we only have 5 generators,
                # the product of a non-backtracking word is determined by the word.
                # We can enumerate all non-backtracking words of length 3 to g-1 and check.
                # Number of such words: sum_{l=3}^{g-1} 5*4^(l-1)
                # For g=9: sum_{l=3}^{8} 5*4^(l-1) = 5*(4^2+4^3+4^4+4^5+4^6+4^7) = 5*21840 = 109200
                # For each word, compute the product (O(l) multiplications) and check if identity.
                # Total operations: ~109200 * 8 * O(1) = ~873600 operations. Very fast.
                
                # But we can do better with incremental computation.
                # Let's use DFS with backtracking.
                
                def dfs_walk(current_prod, last_gen, length):
                    """Enumerate non-backtracking walks from current state."""
                    if length >= g:
                        return False
                    if length >= 3 and current_prod == identity:
                        return True  # Found a short closed non-backtracking walk
                    # Try extending
                    for s_idx in range(5):
                        if s_idx == last_gen:
                            continue
                        new_prod = mat_mult(current_prod, involutions[S[s_idx]])
                        if dfs_walk(new_prod, s_idx, length + 1):
                            return True
                    return False
                
                # Start: identity, no last gen
                # But we need to avoid length 1 and 2 being counted as closed.
                # Length 1: s != id always, so not closed.
                # Length 2: s1*s2 = id iff s2=s1, but s2!=s1 required, so not closed.
                # So we can start DFS at length 0 and check from length 3 onward.
                
                # To avoid deep recursion, let's do iterative level-by-level.
                # Level 0: [(identity, -1)]
                # Level 1: [(involutions[S[i]], i) for i in range(5)]
                # Level l+1: for each (p, last) in level l, for each i != last: (p * involutions[S[i]], i)
                # Check at level l >= 3: any (identity, i) in level l?
                
                # Use sets for deduplication (though in a group, different words can give same product)
                # Actually we DON'T want to deduplicate because different words with same product
                # and same last gen are equivalent for future extensions.
                # So deduplication is fine and helpful.
                
                if time.time() > deadline - 2:
                    return False
                
                # Level 0
                current_level = {(identity, -1)}
                
                for length in range(1, g):
                    if time.time() > deadline - 1:
                        return False
                    next_level = set()
                    for prod, last in current_level:
                        for s_idx in range(5):
                            if s_idx == last:
                                continue
                            new_prod = mat_mult(prod, involutions[S[s_idx]])
                            next_level.add((new_prod, s_idx))
                    
                    # Check for closed walks
                    if length >= 3:
                        for prod, last in next_level:
                            if prod == identity:
                                return False  # Found short cycle
                    # Also check: even at length 2, if we somehow get identity (shouldn't happen)
                    # But let's be safe
                    if length >= 1:
                        for prod, last in next_level:
                            if prod == identity and length >= 3:
                                return False
                    
                    current_level = next_level
                    if not current_level:
                        break
                
                return True

            # Search for good S
            random.seed(42)
            found = False

            # Phase 1: Random search with quick rejection
            max_trials = 20000
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                S_indices = random.sample(range(n_inv), 5)

                # Quick 4-cycle check
                ok = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        if pair_is_inv[S_indices[i]][S_indices[j]]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                # Quick 6-cycle check
                ok = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        for l in range(j + 1, 5):
                            if triple_is_inv[S_indices[i]][S_indices[j]][S_indices[l]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                # Full girth check via non-backtracking walk enumeration
                if check_girth_nb(S_indices):
                    edges = []
                    for u in range(n_vertices):
                        for si in S_indices:
                            v = prod_table[u, si]
                            if u < v:
                                edges.append((u, v))
                    return n_vertices, edges

            # Phase 2: Systematic search with local optimization
            for attempt in range(500):
                if time.time() > deadline - 5:
                    break
                # Generate a set that passes 4-cycle check
                S = None
                for _ in range(100):
                    S = random.sample(range(n_inv), 5)
                    if all(not pair_is_inv[S[i]][S[j]] for i in range(5) for j in range(i + 1, 5)):
                        break
                if S is None:
                    continue

                # Try local search: swap one element at a time
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    if check_girth_nb(S):
                        edges = []
                        for u in range(n_vertices):
                            for si in S:
                                v = prod_table[u, si]
                                if u < v:
                                    edges.append((u, v))
                        return n_vertices, edges

                    # Try swapping one element
                    pos = random.randint(0, 4)
                    old = S[pos]
                    candidates = [x for x in range(n_inv) if x not in S]
                    random.shuffle(candidates)
                    for new in candidates[:30]:
                        S[pos] = new
                        # Quick check
                        ok = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if pair_is_inv[S[i]][S[j]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            ok = True
                            for i in range(5):
                                for j in range(i + 1, 5):
                                    for l in range(j + 1, 5):
                                        if triple_is_inv[S[i]][S[j]][S[l]]:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                        if ok and check_girth_nb(S):
                            edges = []
                            for u in range(n_vertices):
                                for si in S:
                                    v = prod_table[u, si]
                                    if u < v:
                                        edges.append((u, v))
                            return n_vertices, edges
                        S[pos] = old

        except Exception:
            pass

    return luw_fallback()
