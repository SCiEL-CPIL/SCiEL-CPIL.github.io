import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Fallback: LUW construction
    def luw_construct():
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

    # Try PSL(2, 13) Cayley graph
    if k == 5 and g == 9:
        try:
            q = 13
            # PSL(2, q) elements as (a, b, c, d)
            # Identity
            I = (1, 0, 0, 1)
            
            def mul(u, v):
                a1, b1, c1, d1 = u
                a2, b2, c2, d2 = v
                return ((a1*a2 + b1*c2) % q, 
                        (a1*b2 + b1*d2) % q, 
                        (c1*a2 + d1*c2) % q, 
                        (c1*b2 + d1*d2) % q)
            
            def inv(u):
                a, b, c, d = u
                return (d % q, (-b) % q, (-c) % q, a % q)
            
            # Generate all elements of PSL(2, 13)
            # Note: PSL(2, q) has order q(q^2-1)/2
            # We can generate them by iterating a, b, c, d such that ad-bc = 1 or -1 (and identifying inverses)
            # Actually, easier to just generate SL(2, q) and quotient by +/- I.
            # But for girth, we just need the group structure.
            # Let's collect elements in a list and map to indices.
            
            elements = []
            elem_to_idx = {}
            
            # Generate SL(2, q)
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # ad - bc = 1 => ad = 1 + bc => d = (1+bc) * a^-1 if a != 0
                        if a != 0:
                            # a_inv = pow(a, q-2, q)
                            a_inv = pow(a, q-2, q)
                            d = (1 + b*c) * a_inv % q
                            elem = (a, b, c, d)
                            if elem not in elem_to_idx:
                                elem_to_idx[elem] = len(elements)
                                elements.append(elem)
                        else:
                            # a=0 => -bc = 1 => bc = -1. b != 0, c != 0.
                            # d is free? No, ad-bc=1 => -bc=1. d can be anything?
                            # Wait, if a=0, then -bc=1. d is not constrained by determinant=1?
                            # Determinant is ad-bc. If a=0, det = -bc.
                            # So if a=0, we need bc = -1.
                            # d is free? No, the matrix is (0, b, c, d).
                            # Det = -bc. We need -bc = 1.
                            # So b, c must be inverses of -1.
                            # d can be any value?
                            # Yes, if a=0, the first column is (0, c).
                            # The determinant is 0*d - b*c = -bc.
                            # So d is free.
                            if b != 0:
                                c = (-1) * pow(b, q-2, q) % q
                                for d in range(q):
                                    elem = (0, b, c, d)
                                    if elem not in elem_to_idx:
                                        elem_to_idx[elem] = len(elements)
                                        elements.append(elem)
            
            # Now we have SL(2, q). PSL(2, q) is SL(2, q) / {I, -I}.
            # We can just use SL(2, q) and check if girth is maintained?
            # Actually, if we use SL(2, q), the graph is a covering of the PSL graph?
            # No, we want a graph on |PSL| vertices.
            # Let's just use the elements of SL(2, q) as vertices, but identify u and -u?
            # That's complicated.
            # Let's just use the list of elements from SL(2, q) and divide by 2?
            # No, let's just pick a representative for each pair {u, -u}.
            
            psl_elements = []
            psl_to_idx = {}
            seen = set()
            for e in elements:
                if e in seen or inv(e) in seen:
                    continue
                # Add e
                psl_elements.append(e)
                psl_to_idx[e] = len(psl_elements)
                seen.add(e)
                seen.add(inv(e))
            
            n_psl = len(psl_elements)
            # n_psl should be 1092
            
            # Search for generators
            # We need a set S of 5 elements in PSL such that the Cayley graph has girth >= 9.
            # S must be symmetric.
            
            # Precompute multiplication table for speed? 1092^2 is ~1M. Feasible.
            # But we can just compute on the fly for BFS.
            
            def get_girth(S):
                # BFS from identity
                # S is a list of indices in psl_elements
                # We need to map element to index
                # Identity index
                I_idx = psl_to_idx[I]
                
                visited = [-1] * n_psl
                visited[I_idx] = 0
                queue = [(I_idx, 0)]
                
                # We need to check for cycles.
                # In a Cayley graph, a cycle of length L corresponds to a word of length L equal to 1.
                # BFS from 1, if we reach a neighbor of a visited node with depth < current depth?
                # Standard girth BFS:
                # dist[v] = shortest distance from 1 to v.
                # If we find an edge (u, v) such that dist[u] + 1 + dist[v] < current_girth?
                # No, that's for general graphs.
                # For Cayley graph, girth is min length of word in S equal to 1.
                # We can just do BFS and track the "parent" to detect cycles?
                # Actually, since it's a Cayley graph, we can just check if any word of length < 9 is 1.
                # But BFS is efficient.
                
                # Let's use BFS to find the shortest path from 1 to 1 (cycle).
                # But we must avoid backtracking.
                # Non-backtracking walks.
                
                # Simpler: Just check all words of length 1, 2, ..., 8.
                # Number of words is 5^8 = 390,625.
                # For each word, compute the product.
                # This is fast enough.
                
                S_elems = [psl_elements[i] for i in S]
                
                for length in range(1, 9):
                    # Generate all words of this length
                    # Recursive generation
                    words = [(None,)] * length
                    # Use iterative expansion
                    current_words = [I]
                    for _ in range(length):
                        next_words = []
                        for w in current_words:
                            for s in S_elems:
                                next_words.append(mul(w, s))
                        current_words = next_words
                    
                    for w in current_words:
                        if w == I:
                            return length
                return 9 # Or higher
                
            # Random search for S
            # Pick 2 pairs and 1 involution
            # Involution: e = inv(e) => e^2 = I.
            # In PSL(2, 13), involutions exist.
            
            involutions = [e for e in psl_elements if mul(e, e) == I]
            
            found = False
            attempts = 0
            while time.time() < deadline - 5 and attempts < 200:
                attempts += 1
                # Pick 2 random non-involutions
                # Filter out I and involutions
                non_inv = [e for e in psl_elements if e != I and mul(e, e) != I]
                
                u1 = random.choice(non_inv)
                u2 = random.choice(non_inv)
                w = random.choice(involutions)
                
                S_elems = [u1, inv(u1), u2, inv(u2), w]
                
                # Check if S has 5 distinct elements
                if len(set(S_elems)) != 5:
                    continue
                
                # Check if I is in S
                if I in S_elems:
                    continue
                
                # Map to indices
                S_indices = [psl_to_idx[e] for e in S_elems]
                
                # Check girth
                girth = get_girth(S_indices)
                
                if girth >= 9:
                    # Construct the graph
                    edges = []
                    for i, u in enumerate(psl_elements):
                        for j in S_indices:
                            v = psl_elements[j]
                            prod = mul(u, v)
                            v_idx = psl_to_idx[prod]
                            if i < v_idx:
                                edges.append((i, v_idx))
                    return n_psl, edges
                found = True # Just to track
            
        except Exception as e:
            pass
    
    # Fallback to LUW
    return luw_construct()
