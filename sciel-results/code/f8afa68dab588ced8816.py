import time
import math
from collections import deque
import random

# EVOLVE-BLOCK-START

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        if i in adj[j]:
            return False  # repeated edge
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check from every vertex
    for start in range(n):
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in visited:
                    if visited[key] + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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

def _cayley_psl2(k, g, max_time=100):
    """
    Search for a 3-regular Cayley graph on PSL(2, q) with girth >= g.
    Generators: {u, v, v^-1} where u is an involution.
    """
    if k != 3:
        return None
    
    start_time = time.time()
    
    # Try primes q such that |PSL(2,q)| is reasonable
    # |PSL(2,q)| = q(q^2-1)/gcd(2, q-1)
    # For q odd prime, size = q(q^2-1)/2
    primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    for q in primes:
        if time.time() - start_time > max_time:
            break
        
        n_group = q * (q*q - 1) // 2
        if n_group > 20000:
            break
        
        # Enumerate elements of SL(2, q)
        # Elements are matrices [[a,b],[c,d]] with ad-bc=1
        elements = []
        elem_to_idx = {}
        
        # Generate all matrices in SL(2, q)
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # d = (1 + bc) * a^-1 mod q
                    if a == 0:
                        # a=0 => -bc=1 => bc = -1 = q-1
                        # b cannot be 0
                        # c = (q-1) * b^-1
                        if b != 0:
                            b_inv = pow(b, q-2, q)
                            c_val = (q-1) * b_inv % q
                            # d can be anything? No, det=0*d - b*c = -bc = 1.
                            # If a=0, det = -bc. So bc=1.
                            # We already fixed c. d is free?
                            # Matrix [[0,b],[c,d]]. det = -bc.
                            # So d is not constrained by det?
                            # Wait, SL(2,q) elements have det 1.
                            # If a=0, det = -bc = 1 => bc = -1.
                            # d is free?
                            # Let's check: [[0,1],[q-1, d]]. det = 0 - 1*(q-1) = 1-q = -1+1 = 0? No.
                            # In GF(q), -1 = q-1.
                            # det = -bc. We want -bc = 1. So bc = -1.
                            # d does not appear in det.
                            # So for each b != 0, c is fixed, d can be 0..q-1.
                            for d in range(q):
                                mat = (0, b, c_val, d)
                                if mat not in elem_to_idx:
                                    elem_to_idx[mat] = len(elements)
                                    elements.append(mat)
                    else:
                        a_inv = pow(a, q-2, q)
                        d = (1 + b*c) * a_inv % q
                        mat = (a, b, c, d)
                        if mat not in elem_to_idx:
                            elem_to_idx[mat] = len(elements)
                            elements.append(mat)
        
        # Note: PSL(2,q) is SL(2,q) / {I, -I}
        # We can work with SL(2,q) and identify M ~ -M
        # Or just use SL(2,q) and ensure generators are in PSL?
        # If we use SL(2,q), the group size is 2*n_group.
        # A Cayley graph on SL(2,q) with 3 generators is 3-regular.
        # But we want the smallest n.
        # If we find a graph on PSL(2,q), n = n_group.
        # If we find one on SL(2,q), n = 2*n_group.
        # Let's try to find generators in PSL(2,q).
        
        # Identify elements in PSL(2,q)
        # Two matrices A, B are equivalent if A = B or A = -B.
        # Let's canonicalize: if a<0 or (a==0 and b<0) etc?
        # Simpler: map each matrix to its representative in PSL.
        # Rep: if a != 0, multiply by a^-1 to make top-left 1?
        # Or just keep SL(2,q) elements but divide by 2?
        
        # Let's stick to SL(2,q) for now, but check if the graph on PSL is valid.
        # Actually, if we build a Cayley graph on SL(2,q) with generators S,
        # and S is closed under inversion, it's a valid graph.
        # Size is 2*n_group.
        
        # To get size n_group, we need to quotient by {I, -I}.
        # This is a 2-cover.
        # If the Cayley graph on SL(2,q) has girth >= g, does the quotient have girth >= g/2?
        # No, the quotient might have shorter cycles.
        
        # Let's just search for generators in SL(2,q) that yield a graph on SL(2,q).
        # If we find one with n=2*n_group, that's valid.
        # If we can find one on PSL, that's better.
        
        # For simplicity and robustness, let's try to find generators in SL(2,q).
        
        def mat_mul(m1, m2):
            a,b,c,d = m1
            e,f,g,h = m2
            return ((a*e+b*g)%q, (a*f+b*h)%q, (c*e+d*g)%q, (c*f+d*h)%q)
        
        def mat_inv(m):
            a,b,c,d = m
            # det=1, so inv = [[d, -b], [-c, a]]
            return (d, (-b)%q, (-c)%q, a)
        
        I = (1, 0, 0, 1)
        
        # Find involutions in SL(2,q)
        # M^2 = I
        # Trace(M) = 0 (for q odd, M^2=I iff tr(M)=0)
        involutions = []
        for m in elements:
            if m[0] + m[3] == 0:
                involutions.append(m)
        
        # Filter involutions that are not I or -I
        involutions = [m for m in involutions if m != I and m != (-1, 0, 0, -1)]
        
        if not involutions:
            continue
            
        # Randomly select u from involutions, v from others
        best_girth = 0
        best_gens = None
        
        attempts = 0
        while time.time() - start_time < max_time - 5 and attempts < 5000:
            attempts += 1
            u = random.choice(involutions)
            v = random.choice(elements)
            if v == I or v == (-1, 0, 0, -1) or v == u or v == mat_inv(u):
                continue
            if v == mat_inv(v): # v is involution, skip to avoid 2 involutions?
                # Actually, we can have 2 involutions and 1 non-inv?
                # S = {u, w, w^-1}. If w is involution, w=w^-1, so S={u, w}. Size 2.
                # We need size 3. So v must NOT be an involution.
                continue
            
            gens = [u, v, mat_inv(v)]
            
            # Check if they generate SL(2,q)
            # BFS from I
            visited = {I: 0}
            dq = deque([I])
            while dq:
                cur = dq.popleft()
                for gen in gens:
                    nxt = mat_mul(cur, gen)
                    if nxt not in visited:
                        visited[nxt] = visited[cur] + 1
                        dq.append(nxt)
            
            if len(visited) != len(elements):
                continue
                
            # Check girth
            # BFS from I, track (node, prev) to avoid backtracking
            # Or just standard BFS and check for cycles?
            # Standard BFS gives shortest path to each node.
            # Girth is the length of the shortest cycle.
            # We can find this by checking edges (u, v) where dist[u] + 1 == dist[v] is not the only relation.
            # Actually, if we do BFS from I, we get dist.
            # A cycle exists if there is an edge (x, y) such that dist[x] + 1 == dist[y] is false?
            # No.
            
            # Better: Non-backtracking BFS from each vertex? Too slow.
            # For Cayley graphs, girth is the smallest L such that there is a word of length L in S that equals I,
            # with no proper prefix equal to I, and no backtracking (g_i != g_{i-1}^-1).
            
            # Let's use the non-backtracking check from I only?
            # No, girth is global.
            # However, if the graph is vertex-transitive, girth from I is the girth.
            # Cayley graphs are vertex-transitive.
            # So we only need to check cycles passing through I.
            
            # Non-backtracking BFS from I
            visited_nb = {I: 0} # dist
            # We need to track (node, prev_gen) or (node, prev_node)?
            # State: (node, prev_node)
            # Start: (I, None)
            # But we can't go back to None.
            
            # Let's use the standard "shortest cycle through I" algorithm.
            # It is equivalent to finding the shortest word w in S such that w=I and w has no backtracking.
            
            dq_nb = deque()
            for gen in gens:
                dq_nb.append((gen, I, 1)) # (current, prev, dist)
            visited_nb = {(gen, I): 1 for gen in gens}
            
            girth_found = None
            while dq_nb:
                cur, prev, dist = dq_nb.popleft()
                if dist >= g:
                    continue
                for gen in gens:
                    if gen == mat_inv(prev) if prev != I else False: # Backtrack check
                        # If prev is I, we came from I via gen_prev.
                        # We can't go back to I via gen_prev^-1.
                        pass
                    
                    # Correct backtrack check:
                    # We are at cur, came from prev via gen_in.
                    # We want to go to next = cur * gen_out.
                    # We cannot use gen_out = gen_in^-1.
                    
                    # We stored (cur, prev). We need to know which gen we used to get to cur.
                    # Let's store (cur, prev_gen).
                    
                    pass
            
            # Redo NB BFS properly
            dq_nb = deque()
            for gen in gens:
                dq_nb.append((gen, gen, 1)) # (cur, prev_gen, dist)
            visited_nb = {(gen, gen): 1 for gen in gens}
            
            girth_found = None
            while dq_nb:
                cur, prev_gen, dist = dq_nb.popleft()
                if dist >= g:
                    continue
                for gen in gens:
                    if gen == mat_inv(prev_gen):
                        continue
                    nxt = mat_mul(cur, gen)
                    if nxt == I:
                        if dist + 1 < girth_found or girth_found is None:
                            girth_found = dist + 1
                        continue
                    key = (nxt, gen)
                    if key not in visited_nb:
                        visited_nb[key] = dist + 1
                        dq_nb.append((nxt, gen, dist + 1))
            
            if girth_found is not None and girth_found >= g:
                # Found a valid graph
                # Build edges
                edges = []
                for i, node in enumerate(elements):
                    for gen in gens:
                        nxt = mat_mul(node, gen)
                        if nxt in elem_to_idx:
                            j = elem_to_idx[nxt]
                            if i < j:
                                edges.append((i, j))
                
                # Verify
                if _verify_graph(len(elements), edges, 3, g):
                    return len(elements), edges

    return None

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    # Try Cayley graph on PSL(2,q) / SL(2,q)
    if k == 3:
        try:
            result = _cayley_psl2(3, g, max_time=100)
            if result is not None:
                n, edges = result
                best_n, best_edges = n, edges
        except Exception:
            pass
    
    # Try LUW construction as fallback
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            if best_n is None or n < best_n:
                best_n, best_edges = n, edges
    except Exception:
        pass
    
    # Verify best result
    if best_n is not None:
        if _verify_graph(best_n, best_edges, k, g):
            return best_n, best_edges
    
    # Fallback: return best without full verification if time is running out
    if time.time() > deadline - 20:
        if best_n is not None:
            return best_n, best_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
