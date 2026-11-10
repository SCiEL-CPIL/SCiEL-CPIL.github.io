"""Construct a 3-regular graph of girth 16 via Cayley graph on SL(2, q)."""
import time
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, g=16: use Cayley graph on SL(2, 16) with appropriate generators.
    """
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # For the specific task: k=3, g=16
    # Use SL(2, q) where q is a prime power
    # SL(2, 16) has 16^3 - 16 = 4080 elements
    # We need to find 3 generators that give girth >= 16
    
    # SL(2, q) construction
    # Elements are 2x2 matrices with determinant 1 over GF(q)
    
    def gf_pow(a, p, q):
        """Fast exponentiation in GF(q) for prime q."""
        result = 1
        base = a % q
        while p:
            if p & 1:
                result = result * base % q
            base = base * base % q
            p >>= 1
        return result
    
    def gf_inv(a, q):
        """Inverse in GF(q) for prime q."""
        return gf_pow(a, q - 2, q)
    
    def sl2_elements(q):
        """Generate all elements of SL(2, q) as (a, b, c, d) tuples."""
        elems = []
        # a*d - b*c = 1 mod q
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # d = (1 + b*c) * a^{-1} if a != 0
                    if a == 0:
                        # Then -b*c = 1, so b != 0 and c != 0
                        if b == 0 or c == 0:
                            continue
                        if (b * c) % q != (q - 1) % q:  # -1 mod q
                            continue
                        # d can be anything? No: a=0, so 0*d - b*c = 1 => -bc=1
                        # d is free? No, the constraint is ad-bc=1, with a=0: -bc=1
                        # So d is determined? No, d doesn't appear. Wait:
                        # ad - bc = 0*d - bc = -bc = 1. So bc = -1.
                        # d is free! But that gives q elements for each (b,c) with bc=-1.
                        # Actually for SL(2,q), when a=0, we need -bc=1, and d is arbitrary.
                        for d in range(q):
                            elems.append((0, b, c, d))
                    else:
                        # d = (1 + b*c) * a^{-1}
                        d = (1 + b * c) % q * gf_inv(a, q) % q
                        elems.append((a, b, c, d))
        return elems
    
    def mat_mul(e1, e2, q):
        """Multiply two SL(2,q) elements."""
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        return (
            (a1 * a2 + b1 * c2) % q,
            (a1 * b2 + b1 * d2) % q,
            (c1 * a2 + d1 * c2) % q,
            (c1 * b2 + d1 * d2) % q
        )
    
    def mat_inv(e, q):
        """Inverse in SL(2,q): (a,b,c,d)^{-1} = (d, -b, -c, a)."""
        a, b, c, d = e
        return (d, (-b) % q, (-c) % q, a)
    
    def mat_pow(e, p, q):
        """Fast exponentiation of SL(2,q) element."""
        result = (1, 0, 0, 1)  # identity
        base = e
        while p:
            if p & 1:
                result = mat_mul(result, base, q)
            base = mat_mul(base, base, q)
            p >>= 1
        return result
    
    # Try SL(2, 16) - but 16 is not prime, so we need GF(16)
    # Let's use SL(2, 17) instead (prime), which has 17^3 - 17 = 4896 elements
    # Or SL(2, 13) with 13^3 - 13 = 2197 elements
    # Or SL(2, 11) with 11^3 - 11 = 1320 elements
    
    # Actually, let me think about this more carefully.
    # For a 3-regular Cayley graph on a group G with generators S = {s1, s2, s3}
    # where s_i^{-1} is also in S or we use undirected edges,
    # the girth is the length of the shortest non-trivial word in S that equals identity.
    
    # For k=3, we can use generators {a, b, c} where the Cayley graph has
    # edges from g to g*a, g*b, g*c. If a, b, c are involutions or we handle
    # inverses, we need to be careful.
    
    # Alternative approach: Use the known construction from the literature.
    # The Lazebnik-Ustimenko-Woldar construction gives a 3-regular bipartite graph.
    # For girth 16, we need d such that d+5 >= 16, so d >= 11.
    # With q=3, d=11: the graph has about 3^11 * 2 = 177147 vertices. That's too many.
    
    # Let me try a different approach: use a random 3-regular graph and check girth,
    # or use a more structured construction.
    
    # Actually, for a quick improvement, let me use the LUW construction with smaller parameters
    # and verify the girth is at least 16.
    
    # LUW construction: CD(d, q) where q is prime power, d is odd >= 3
    # Girth >= d + 5
    # For girth 16: d >= 11
    # Number of vertices in the connected component: roughly 2 * (1 + q + q^2 + ... + q^d) / something
    # Actually it's a tree-like structure, so |V| = 2 * (q^d + q^{d-1} + ... + q + 1) / (q+1) * ... 
    # Let me just compute it.
    
    # For q=3, d=11: the number of vertices is 2 * sum_{i=0}^{11} 3^i = 2 * (3^12 - 1)/2 = 3^12 - 1 = 531440
    # That's way too many.
    
    # For q=2, d=11: 2 * sum_{i=0}^{11} 2^i = 2 * (2^12 - 1) = 8190. But q must be >= 3 for the construction to work properly?
    # Actually q=2 might work but the graph might not be 3-regular.
    
    # Let me reconsider. The task says the smallest known is 936 vertices.
    # Let me try to construct a graph with around that size.
    
    # Approach: Use a random 3-regular graph with n vertices and check if girth >= 16.
    # For n ~ 1000, we can do this.
    
    # But random 3-regular graphs typically have girth ~ log_{2}(n) which for n=1000 is about 10.
    # So we need a more structured approach.
    
    # Let me use the Cayley graph approach on a finite group.
    # For SL(2, 16) (over GF(16)), |SL(2,16)| = 16^3 - 16 = 4080.
    # We need to find 3 generators that give girth >= 16.
    
    # Since 16 = 2^4, we need arithmetic in GF(16).
    
    # GF(16) = GF(2)[x]/(x^4 + x + 1)
    # Let's represent elements as integers 0-15, with multiplication via polynomial mod x^4+x+1.
    
    def gf16_mult(a, b):
        """Multiply in GF(16) = GF(2)[x]/(x^4 + x + 1)."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x10:  # x^4
                a ^= 0x03  # x^4 + x + 1, so x^4 = x + 1
        return result & 0x0F
    
    def gf16_inv(a):
        """Inverse in GF(16)."""
        if a == 0:
            return 0
        # a^15 = 1 in GF(16)*
        return gf16_mult(gf16_mult(a, a), gf16_mult(gf16_mult(a, a), gf16_mult(a, a)))
        # Actually: a^14 * a = 1, so inv = a^14
        # a^2, a^4, a^8, a^14 = a^8 * a^4 * a^2
        a2 = gf16_mult(a, a)
        a4 = gf16_mult(a2, a2)
        a8 = gf16_mult(a4, a4)
        return gf16_mult(gf16_mult(a8, a4), a2)
    
    def sl2_16_elements():
        """All elements of SL(2, 16) as (a, b, c, d) with ad - bc = 1 in GF(16)."""
        elems = []
        for a in range(16):
            for b in range(16):
                for c in range(16):
                    # ad - bc = 1 => ad = 1 + bc
                    # If a != 0: d = (1 + bc) * a^{-1}
                    # If a == 0: -bc = 1 => bc = 15 (since -1 = 15 in GF(16))
                    if a == 0:
                        # bc must equal 15 (which is -1 in GF(16))
                        if b == 0 or c == 0:
                            continue
                        if gf16_mult(b, c) != 15:
                            continue
                        for d in range(16):
                            elems.append((0, b, c, d))
                    else:
                        bc = gf16_mult(b, c)
                        rhs = (1 + bc) & 0x0F  # 1 + bc in GF(16)
                        d = gf16_mult(rhs, gf16_inv(a))
                        elems.append((a, b, c, d))
        return elems
    
    def sl2_16_mul(e1, e2):
        """Multiply two SL(2,16) elements."""
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        return (
            (gf16_mult(a1, a2) ^ gf16_mult(b1, c2)) & 0x0F,
            (gf16_mult(a1, b2) ^ gf16_mult(b1, d2)) & 0x0F,
            (gf16_mult(c1, a2) ^ gf16_mult(d1, c2)) & 0x0F,
            (gf16_mult(c1, b2) ^ gf16_mult(d1, d2)) & 0x0F
        )
    
    def sl2_16_inv(e):
        """Inverse in SL(2,16)."""
        a, b, c, d = e
        return (d, b ^ 0x0F, c ^ 0x0F, a)  # -b = b XOR 15 in GF(16)
    
    def sl2_16_pow(e, p):
        """Fast exponentiation."""
        result = (1, 0, 0, 1)
        base = e
        while p:
            if p & 1:
                result = sl2_16_mul(result, base)
            base = sl2_16_mul(base, base)
            p >>= 1
        return result
    
    # Generate SL(2, 16)
    group = sl2_16_elements()
    n = len(group)
    elem_to_idx = {e: i for i, e in enumerate(group)}
    
    # Find generators: we need 3 elements s1, s2, s3 such that the Cayley graph
    # has girth >= 16.
    # The girth is the shortest non-trivial word in {s1, s2, s3, s1^{-1}, s2^{-1}, s3^{-1}}
    # that equals the identity.
    
    # Strategy: pick random triples and check girth.
    # For efficiency, precompute powers up to some depth.
    
    identity = (1, 0, 0, 1)
    
    import random
    random.seed(42)
    
    # Precompute all group elements indexed
    # For girth checking, we need to check all words of length up to 15
    # that don't immediately backtrack.
    
    # Number of non-backtracking walks of length L in a 3-regular graph from a vertex:
    # 3 * 2^(L-1)
    # For L=15: 3 * 2^14 = 49152 per starting vertex. With 4080 vertices, that's ~200M checks.
    # Too slow in Python.
    
    # Alternative: just check words up to length 15 from the identity.
    # Number of non-backtracking words of length L: 3 * 2^(L-1)
    # Total up to length 15: sum_{L=1}^{15} 3*2^(L-1) = 3*(2^15 - 1) = 98301
    # That's manageable.
    
    def check_girth(generators, max_len=15):
        """Check if the Cayley graph with given generators has girth > max_len."""
        # generators: list of 3 group elements
        # Check all non-backtracking words of length up to max_len
        # A word is non-backtracking if no two consecutive letters are inverses.
        
        # Map generator to index: 0, 1, 2 and their inverses: 0', 1', 2'
        # Represent letters as: 0,1,2 for s0,s1,s2 and 3,4,5 for s0^{-1},s1^{-1},s2^{-1}
        
        gens = generators
        inv_gens = [sl2_16_inv(g) for g in gens]
        
        # letter_to_elem: 0->s0, 1->s1, 2->s2, 3->s0^{-1}, 4->s1^{-1}, 5->s2^{-1}
        letter_to_elem = gens + inv_gens
        # inverse letter: 0<->3, 1<->4, 2<->5
        inv_letter = [3, 4, 5, 0, 1, 2]
        
        # BFS over non-backtracking words
        # State: (current_group_element, last_letter)
        # We want to find if any word of length <= max_len equals identity
        # with no immediate backtracking.
        
        # Start: identity, no last letter
        # After first letter: (s_i, i) for i in 0,1,2,3,4,5
        # But we can restrict: since the graph is undirected, we can assume
        # the first letter is one of s0, s1, s2 (not inverses) by symmetry.
        
        visited = set()
        # Queue: (element, last_letter, length)
        queue = []
        
        # First step
        for i in range(6):
            e = letter_to_elem[i]
            if e == identity:
                return False  # generator is identity
            queue.append((e, i, 1))
            visited.add((e, i))
        
        for length in range(1, max_len):
            new_queue = []
            for e, last, l in queue:
                if e == identity and l > 1:
                    return False
                for i in range(6):
                    if i == inv_letter[last]:
                        continue  # backtrack
                    ne = sl2_16_mul(e, letter_to_elem[i])
                    state = (ne, i)
                    if state not in visited:
                        visited.add(state)
                        new_queue.append((ne, i, l + 1))
            queue = new_queue
            if not queue:
                break
            if time.time() > deadline - 10:
                break
        
        return True
    
    # Try random generator sets
    best_n = None
    best_edges = None
    
    # For a 3-regular Cayley graph on group G with generators S = {s1, s2, s3},
    # edges are (g, g*s_i) for all g and i=1,2,3.
    # But this gives a directed graph. For undirected, we need s_i^{-1} in S or
    # we add both directions.
    # Standard: if S is closed under inverses, or we define edges as {g, g*s_i} for each g.
    # For 3-regular: each vertex g connects to g*s1, g*s2, g*s3.
    # This is 3-regular if s1, s2, s3 are distinct and none is identity.
    
    # Let's try random triples from the group.
    
    # Filter out identity and elements of small order
    candidates = [e for e in group if e != identity]
    
    # Try some random triples
    attempts = 0
    max_attempts = 50000
    
    while attempts < max_attempts and time.time() < deadline - 20:
        s1 = candidates[random.randint(0, len(candidates) - 1)]
        s2 = candidates[random.randint(0, len(candidates) - 1)]
        s3 = candidates[random.randint(0, len(candidates) - 1)]
        attempts += 1
        
        # Check distinctness
        if s1 == s2 or s2 == s3 or s1 == s3:
            continue
        if s1 == identity or s2 == identity or s3 == identity:
            continue
        
        # Check that s1, s2, s3 don't have small orders (order 2 is ok, but let's avoid order 1)
        # Actually order 2 elements are fine (involutions)
        
        gens = [s1, s2, s3]
        
        if check_girth(gens, max_len=15):
            # Found a valid set of generators!
            # Build the graph
            edges = []
            for i, g in enumerate(group):
                for s in gens:
                    gs = sl2_16_mul(g, s)
                    j = elem_to_idx[gs]
                    if i < j:
                        edges.append((i, j))
                    # Actually for undirected, we add edge (i, j) once
                    # But we need to make sure we don't add duplicates
                    # Since the graph is undirected and 3-regular, each vertex has 3 neighbors
                    # Edge (i,j) is added when we process i and find j as neighbor,
                    # and when we process j and find i as neighbor.
                    # To avoid duplicates, only add if i < j.
            
            # Verify it's 3-regular
            from collections import defaultdict
            deg = defaultdict(int)
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(deg[i] == 3 for i in range(n)):
                return n, edges
            else:
                # Something went wrong, continue
                continue
    
    # Fallback: use the LUW construction
    # This is the original approach, which is guaranteed to work but gives a large graph
    
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

# EVOLVE-BLOCK-END
