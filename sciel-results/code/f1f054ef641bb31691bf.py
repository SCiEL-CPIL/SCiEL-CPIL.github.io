import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

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

    def generalized_hexagon():
        """Construct the incidence graph of the generalized hexagon of order (4,4).
        Points = PG(2,4) elements (121), Lines = PG(2,4)* elements (121).
        Total vertices = 242. Each point is collinear with 5 lines, each line with 5 points.
        Girth = 12 (>= 9)."""
        q = 4
        # Build all points of PG(2,4) as (x, y, z) with z=1 or (x, 0, 1) for infinity
        # Actually let's use the standard representation:
        # Points: (a, b) in F_4^2 (16), plus points at infinity (1, t) for t in F_4 (4), plus (1, 0, 0) (1) = 21? No.
        # PG(2,4) has (4^3-1)/(4-1) = 63/3 = 21 points. Wait, that's wrong.
        # |PG(2,q)| = (q^3-1)/(q-1) = q^2+q+1 = 16+4+1 = 21 for q=4.
        # Hmm, but the hexagon of order (4,4) has (q^3-1)(q^3+1)/((q-1)(q+1)) vertices for the incidence graph?
        # Actually for a generalized hexagon of order (s,t):
        # Number of points = (s+1)(st+1) = 5*(16+1) = 85? No.
        # For a generalized n-gon of order (s,t):
        # Number of points = (s+1)(s^2 t^2 + s t + 1) for hexagon? 
        # Let me use the known result: the generalized hexagon of order (4,4) has 330 vertices total.
        # 330 = 2 * 165. Points = 165, Lines = 165? Or points = 121, lines = 121, total = 242?
        # Actually 330 = 2*3*5*11. The known hexagon H(4,4) has:
        # Number of points = (q+1)(q^3-1)/(q-1) ... let me just use the known structure.
        # 
        # The generalized hexagon of order (s,t) with s=t=4:
        # |Points| = (s+1)(st+1) * something... 
        # Known: for the hexagon of order (4,4), there are 330 vertices in the incidence graph.
        # Each vertex has degree 5 (since s+1 = t+1 = 5).
        # 
        # Let me construct it via the standard method using the group of automorphisms or
        # the explicit coordinate construction.
        
        # Standard construction: Points are triples (a, b, c) in F_q^3 with certain conditions,
        # or use the fact that it's a Cayley graph.
        
        # Alternative: Use the known fact that the generalized hexagon of order (4,4) 
        # can be constructed from the projective line over F_4 with a specific relation.
        
        # Let me use the explicit construction from the theory:
        # The generalized hexagon of order (s,t) where s=t=q has a representation using
        # the field F_q. Points are pairs (i, (x,y)) where i is in {0,...,q^2+q} and (x,y) in F_q^2,
        # but this gets complicated.
        
        # Simpler: Use the known Cayley graph representation.
        # The generalized hexagon of order (4,4) is a Cayley graph of the group of order 165
        # (the simple group PSL(2,11) has order 660, not right).
        # Actually 330/2 = 165. The group of order 165 is not simple (165 = 3*5*11).
        
        # Let me just use a direct construction. The incidence graph of the generalized hexagon
        # of order (4,4) can be built as follows:
        # 
        # Use the representation where points are (i, j) with i in Z_5, j in Z_33? No.
        # 
        # Actually, let me use the well-known construction via the (5,3)-cage or similar.
        # 
        # For a 5-regular graph with girth 12 on 330 vertices, let me use the following approach:
        # The generalized hexagon H(4,4) has a well-known representation. Let me construct it
        # using the standard algebraic method.
        
        # Field F_4 = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
        # Represent F_4 elements as integers 0,1,2,3 with:
        # 0=0, 1=1, 2=alpha, 3=alpha^2
        # Addition: 0+0=0, 0+1=1, 0+2=2, 0+3=3, 1+1=0, 1+2=3, 1+3=2, 2+2=0, 2+3=1, 3+3=0
        # Multiplication: 0*x=0, 1*x=x, 2*2=3, 2*3=2, 3*3=2 (since alpha^2*alpha^2 = alpha^4 = alpha)
        
        f4_add = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]], dtype=np.int32)
        f4_mul = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,2],[0,3,2,2]], dtype=np.int32)
        
        # Inverse in F_4: 0->0, 1->1, 2->3, 3->2
        f4_inv = np.array([0, 1, 3, 2], dtype=np.int32)
        
        # Points of PG(2,4): represented as (x, y) in F_4^2 (16 affine points)
        # plus 4 points at infinity (one for each slope in F_4) plus 1 point at infinity for vertical = 21
        # Wait, PG(2,4) has 21 points. But we need 165 points for the hexagon.
        # 
        # I think I'm confusing myself. Let me look at this differently.
        # 
        # The generalized hexagon of order (4,4) has:
        # - 330 vertices (points + lines in the incidence graph)
        # - Each vertex has degree 5
        # - Girth 12
        # 
        # This is a known graph. Let me construct it using a different method.
        # 
        # Method: Use the fact that it's a (5,12)-cage. The smallest known 5-regular graph 
        # with girth 12 has 330 vertices. Let me construct it using a voltage graph or
        # a group-based construction.
        
        # Actually, let me just use a simpler approach: construct a 5-regular graph on 330 
        # vertices using a known explicit construction.
        
        # The generalized hexagon of order (4,4) can be constructed as follows:
        # Let G be the group with presentation related to the hexagon.
        # 
        # Alternative simple approach: Use the incidence structure directly.
        # In the generalized hexagon of order (s,t) = (4,4):
        # - Each point is incident with s+1 = 5 lines
        # - Each line is incident with t+1 = 5 points  
        # - Number of points = (s+1)(s^2 t^2 + st + 1) = 5*(256+16+1) = 5*273 = 1365? No that's too many.
        # 
        # Correct formula for generalized n-gon of order (s,t):
        # For a hexagon (n=6): number of points = (s+1)(st+1) * (s^2 t^2 + st + 1)/(st+1) ... 
        # 
        # Let me just use the known values: for H(4,4), |V| = 330, so 165 points and 165 lines.
        # 
        # 165 = 3 * 5 * 11. 
        # 
        # Construction via coordinates:
        # Points: (i, j, k) where i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,...,10}
        # That gives 3*5*11 = 165 points.
        # Lines: similar structure.
        # Incidence: two points are on a line if they differ in exactly one coordinate.
        # But that would give degree 5 (change one of 3 coords, but each coord has 5 or 11 choices...).
        # Actually if points are (i,j,k) with i in Z_3, j in Z_5, k in Z_11, and two points are
        # adjacent if they differ in exactly one coordinate, then each point has degree 2+4+10 = 16, not 5.
        
        # Let me try a different approach. Use a circulant-like construction.
        # 
        # For a 5-regular graph on 330 vertices with girth 12, I can use:
        # V = Z_330, edges: (i, i+1), (i, i+2), (i, i+3), (i, i+4), (i, i+5) mod 330?
        # That's a circulant graph C(330, {1,2,3,4,5}). Its girth is likely small.
        
        # Let me use a proper construction. The generalized hexagon of order (4,4) is a well-known
        # object. Let me construct it using the standard method from the literature.
        
        # Standard construction of the generalized hexagon of order (q, q) for q = 4:
        # Points: (a, b, c) in F_q^3 not all zero, modulo scalar multiplication, with c != 0 or c=0 and b != 0.
        # That gives PG(2,q) which has q^2+q+1 = 21 points for q=4. Not 165.
        
        # I think the hexagon of order (4,4) is NOT the same as PG(2,4). Let me reconsider.
        # 
        # The generalized hexagon of order (s,t) has:
        # Number of points = (s+1)(s^2 t^2 + st + 1)
        # For s=t=4: (5)(256+16+1) = 5*273 = 1365. That's too many.
        # 
        # Wait, I think I have the formula wrong. Let me use:
        # For a generalized n-gon of order (s,t):
        # Number of points = (s+1) * (1 + st + (st)^2 + ... + (st)^{n/2-1})
        # For hexagon (n=6): (s+1) * (1 + st + (st)^2) = 5 * (1 + 16 + 256) = 5 * 273 = 1365
        # 
        # That's definitely not 330. So the "generalized hexagon of order (4,4)" with 330 vertices
        # must be something else, or I'm misremembering.
        
        # Let me reconsider. Maybe it's not a generalized hexagon but a different structure.
        # 
        # The 5-regular graph with girth 12 on 330 vertices: this is the (5,12)-cage.
        # The Moore bound for (5,12) is: 2 * sum_{i=0}^{5} 5^i = 2*(1+5+25+125+625+3125) = 2*3906 = 7812? No.
        # Moore bound for even girth g=2m: n >= 2 * sum_{i=0}^{m-1} (k-1)^i = 2 * sum_{i=0}^{5} 4^i = 2*(1+4+16+64+256+1024) = 2*1365 = 2730.
        # 
        # So the Moore bound is 2730, and 330 is WAY below that. That can't be right for girth 12.
        # 
        # I think I'm confusing myself. Let me reconsider what graph has 330 vertices, is 5-regular, and has girth >= 9.
        # 
        # Moore bound for (5,9): g=9 is odd, so m=4, n >= 2*sum_{i=0}^{3} 4^i + 4^4 = 2*(1+4+16+64) + 256 = 2*85 + 256 = 170 + 256 = 426.
        # 
        # So for girth 9, the Moore bound is 426, and the best known is 1116. The current program's 
        # fallback (LUW) probably gives something around 1116 or more.
        # 
        # So I need a 5-regular graph with girth >= 9 on as few vertices as possible, under 1116.
        # 
        # The PSL(2,13) approach in the current code: |PSL(2,13)| = 13*(13^2-1)/2 = 13*168/2 = 1092.
        # So the Cayley graph would have 1092 vertices. If the random search succeeds, it gives n=1092.
        # 
        # Can I do better? The task says the smallest known is 1116, so 1092 would be an improvement!
        # But the current program scores 1.0, meaning the PSL(2,13) search is failing.
        # 
        # Let me debug: the issue is likely in the girth check or the search not finding valid generators.
        # 
        # Let me try a more systematic approach for PSL(2,13):
        # 1. Precompute all involutions
        # 2. Systematically try all 5-element subsets (or a large random sample)
        # 3. Use a faster girth check
        
        # Actually, let me try a completely different approach: use a known explicit construction.
        # 
        # The Hoffman-Singleton graph is (7,5)-cage with 50 vertices. Not helpful.
        # 
        # Let me try the approach of constructing a 5-regular graph with girth 9 using
        # a product of smaller graphs or a voltage graph.
        
        # Actually, let me fix the PSL(2,13) approach. The key issue is that the random 
        # search may not be finding valid generator sets. Let me make it more systematic.
        
        pass  # This function is a placeholder; I'll implement the real construction below
    
    # Main construction: PSL(2,13) Cayley graph with systematic generator search
    if k == 5 and g <= 12:
        try:
            q = 13
            # Build PSL(2,13) elements
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        for d in range(q):
                            det = (a * d - b * c) % q
                            if det != 1:
                                continue
                            sl_elements.append((a, b, c, d))
            
            # PSL(2,13) = SL(2,13) / {I, -I}
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d = e
                key = (a, b, c, d)
                neg_key = (-a % q, -b % q, -c % q, -d % q)
                if key < neg_key:
                    if key not in seen:
                        seen.add(key)
                        psl_elements.append(e)
                else:
                    if neg_key not in seen:
                        seen.add(neg_key)
                        psl_elements.append(neg_key)
            
            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            
            # Precompute multiplication table
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            for i in range(n_vertices):
                a1, b1, c1, d1 = psl_elements[i]
                for j in range(n_vertices):
                    a2, b2, c2, d2 = psl_elements[j]
                    prod = ((a1 * a2 + b1 * c2) % q,
                            (a1 * b2 + b1 * d2) % q,
                            (c1 * a2 + d1 * c2) % q,
                            (c1 * b2 + d1 * d2) % q)
                    mul_table[i, j] = idx_map[prod]
            
            # Find involution indices (elements of order 2, i.e., trace = 0, not identity)
            involution_indices = []
            for i, e in enumerate(psl_elements):
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    involution_indices.append(i)
            
            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)
            
            # Precompute is_involution lookup
            is_involution = np.zeros(n_vertices, dtype=bool)
            for i in involution_indices:
                is_involution[i] = True
            
            # Precompute pair products of involutions
            # inv_prod[i, j] = mul_table[involution_indices[i], involution_indices[j]]
            inv_prod = mul_table[np.ix_(involution_indices, involution_indices)]
            
            # For each pair of involution indices, check if their product is an involution
            # This tells us which pairs form 4-cycles
            pair_is_inv = is_involution[np.vectorize(lambda x: x, otypes=[np.int32])(inv_prod)]
            # Actually, let's just use a lookup
            pair_is_inv = np.zeros((n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(n_inv):
                    pair_is_inv[i, j] = is_involution[inv_prod[i, j]]
            
            # Precompute triple products
            # triple[i, j, l] = mul_table[inv_prod[i,j], involution_indices[l]]
            # Shape: (n_inv, n_inv, n_inv) - this is large (168^3 = 4.7M), but manageable
            # Actually n_inv for PSL(2,13): involutions in PSL(2,13).
            # In PSL(2,q) for q odd, number of involutions = q(q+1)/2 for q ≡ 1 mod 4, or q(q-1)/2 for q ≡ 3 mod 4.
            # q=13 ≡ 1 mod 4, so n_inv = 13*14/2 = 91.
            # 91^3 = 753,571 - manageable.
            
            triple_is_inv = np.zeros((n_inv, n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(n_inv):
                    if i == j:
                        continue
                    for l in range(n_inv):
                        if l == i or l == j:
                            continue
                        prod = mul_table[inv_prod[i, j], involution_indices[l]]
                        triple_is_inv[i, j, l] = is_involution[prod]
            
            # Precompute quad products
            # quad[i, j, l, m] = mul_table[triple_prod, involution_indices[m]]
            # 91^4 = 68.5M - too large to precompute fully.
            # Instead, compute on the fly for the 5 chosen generators.
            
            def check_girth_for_generators(S_idx):
                """Check if 5 involutions S_idx (indices into involution_indices) give girth >= g."""
                # 4-cycle: for all i<j, check if involution_indices[S_idx[i]] * involution_indices[S_idx[j]] is involution
                for i in range(5):
                    for j in range(i + 1, 5):
                        if pair_is_inv[S_idx[i], S_idx[j]]:
                            return False
                
                # 6-cycle: for all distinct i,j,l, check if triple product is involution
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            continue
                        for l in range(5):
                            if l == i or l == j:
                                continue
                            if triple_is_inv[S_idx[i], S_idx[j], S_idx[l]]:
                                return False
                
                # 8-cycle: for all distinct i,j,l,m with i!=j, j!=l, l!=m, m!=i
                # Compute quad product on the fly
                psl_S = involution_indices[S_idx]
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            continue
                        for l in range(5):
                            if l == i or l == j:
                                continue
                            for m in range(5):
                                if m == j or m == l:
                                    continue
                                # Compute s_i * s_j * s_l * s_m
                                tmp1 = mul_table[psl_S[i], psl_S[j]]
                                tmp2 = mul_table[tmp1, psl_S[l]]
                                prod = mul_table[tmp2, psl_S[m]]
                                if is_involution[prod]:
                                    return False
                
                return True
            
            # Search for valid generator sets
            best_edges = None
            np.random.seed(42)
            
            # First, try all 5-subsets from a smaller pool to increase hit rate
            # Sort involutions by some criterion and try combinations
            max_attempts = 200000
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                    
                S_idx = np.random.choice(n_inv, size=5, replace=False)
                
                if not check_girth_for_generators(S_idx):
                    continue
                
                # Check if S generates the group
                psl_S = involution_indices[S_idx]
                visited = np.zeros(n_vertices, dtype=bool)
                visited[0] = True
                queue = [0]
                head = 0
                edges = []
                
                while head < len(queue):
                    if time.time() > deadline - 10:
                        break
                    u = queue[head]
                    head += 1
                    for s in range(5):
                        v = mul_table[u, psl_S[s]]
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            edges.append((u, v))
                
                if not visited.all():
                    continue
                
                # Verify girth with non-backtracking BFS from identity
                dist = np.full(n_vertices, -1, dtype=np.int32)
                dist[0] = 0
                queue_nb = [(0, -1)]
                head_nb = 0
                girth_ok = True
                while head_nb < len(queue_nb):
                    if time.time() > deadline - 8:
                        girth_ok = False
                        break
                    u, prev_s = queue_nb[head_nb]
                    head_nb += 1
                    du = dist[u]
                    if du >= (g + 1) // 2:
                        continue
                    for s in range(5):
                        if s == prev_s:
                            continue
                        v = mul_table[u, psl_S[s]]
                        dv = dist[v]
                        if dv == -1:
                            dist[v] = du + 1
                            queue_nb.append((v, s))
                        else:
                            if du + 1 + dv < g:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    best_edges = edges
                    break
            
            if best_edges is not None:
                return n_vertices, best_edges
        
        except Exception:
            pass
    
    return luw_fallback()
