import math
import time

# Deadline guard: 120s limit, use 110s to be safe
_DEADLINE = 110.0

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    if k != 4:
        return _construct_fallback(k, g)
    
    start_time = time.time()
    
    # For g=10, try known constructions
    if g == 10:
        # Try the Cayley graph on Z_2 x Z_2 x Z_5 x Z_16 (320 vertices)
        result = _try_cayley_z2z2z5z16(start_time)
        if result:
            return result
        
        # Try other abelian group constructions
        result = _try_abelian_groups(start_time)
        if result:
            return result
    
    # Fallback to Lazebnik-Ustimenko-Woldar construction
    return _construct_fallback(k, g)


def _try_cayley_z2z2z5z16(start_time):
    """Try Cayley graph on G = Z_2 x Z_2 x Z_5 x Z_16 with girth >= 10."""
    # Pre-compute all non-backtracking words of length 2 to 9
    # and their exponent pairs (k1, k2) where word = s1^k1 * s2^k2 in free group
    
    exponents = set()
    
    # Generate non-backtracking words over alphabet {1, -1, 2, -2}
    # where 1 = s1, -1 = s1^-1, 2 = s2, -2 = s2^-1
    # Non-backtracking: consecutive symbols not inverses
    
    def generate_words():
        """Generate all non-backtracking words of length 2-9 and return their exponent pairs."""
        # Use iterative approach to avoid recursion depth issues
        words = []
        # Start with single symbols
        for sym in [1, -1, 2, -2]:
            k1 = 1 if sym == 1 else -1 if sym == -1 else 0
            k2 = 1 if sym == 2 else -1 if sym == -2 else 0
            words.append(([sym], k1, k2))
        
        current_length = 1
        while current_length < 9:
            next_words = []
            for word, k1, k2 in words:
                last = word[-1]
                for sym in [1, -1, 2, -2]:
                    if sym == -last:
                        continue  # non-backtracking
                    new_k1 = k1 + (1 if sym == 1 else -1 if sym == -1 else 0)
                    new_k2 = k2 + (1 if sym == 2 else -1 if sym == -2 else 0)
                    new_word = word + [sym]
                    next_words.append((new_word, new_k1, new_k2))
                    if len(new_word) >= 2:
                        exponents.add((new_k1, new_k2))
            words = next_words
            current_length += 1
        
        # Also add length-2 words
        for word, k1, k2 in words:
            if len(word) >= 2:
                exponents.add((k1, k2))
        
        return list(exponents)
    
    exp_list = generate_words()
    
    if time.time() - start_time > _DEADLINE - 5:
        return None
    
    # Group elements: (a, b, c, d) with a,b in 0..1, c in 0..4, d in 0..15
    # Total: 2*2*5*16 = 320
    
    def add_g(e1, e2):
        return ((e1[0]+e2[0])%2, (e1[1]+e2[1])%2, (e1[2]+e2[2])%5, (e1[3]+e2[3])%16)
    
    def neg_g(e):
        return ((-e[0])%2, (-e[1])%2, (-e[2])%5, (-e[3])%16)
    
    def mul_g(k, e):
        """Compute k*e in the group."""
        if k == 0:
            return (0, 0, 0, 0)
        if k == 1:
            return e
        if k == -1:
            return neg_g(e)
        
        if k > 0:
            res = (0, 0, 0, 0)
            for _ in range(k):
                res = add_g(res, e)
            return res
        else:
            return neg_g(mul_g(-k, e))
    
    # Elements of the group
    elements = [(a, b, c, d) for a in range(2) for b in range(2) for c in range(5) for d in range(16)]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Try pairs (s1, s2) as generators
    # For efficiency, we can use symmetry: assume s1 has smaller "norm" than s2
    # But for now, just iterate
    
    for s1 in elements:
        if s1 == (0, 0, 0, 0):
            continue
        if time.time() - start_time > _DEADLINE - 10:
            return None
        
        for s2 in elements:
            if s2 == (0, 0, 0, 0):
                continue
            # Ensure s1 != s2, s1 != -s2 to avoid degenerate cases
            if s1 == s2 or s1 == neg_g(s2):
                continue
            
            # Check girth condition: no non-backtracking closed walk of length < 10
            valid = True
            for k1, k2 in exp_list:
                term1 = mul_g(k1, s1)
                term2 = mul_g(k2, s2)
                res = add_g(term1, term2)
                if res == (0, 0, 0, 0):
                    valid = False
                    break
            
            if valid:
                # Construct edges
                edges = set()
                for i, v in enumerate(elements):
                    n1 = add_g(v, s1)
                    n2 = add_g(v, neg_g(s1))
                    n3 = add_g(v, s2)
                    n4 = add_g(v, neg_g(s2))
                    
                    for n in [n1, n2, n3, n4]:
                        j = elem_to_idx[n]
                        if i < j:
                            edges.add((i, j))
                        else:
                            edges.add((j, i))
                
                return 320, sorted(edges)
    
    return None


def _try_abelian_groups(start_time):
    """Try other abelian group constructions for 4-regular girth-10 graphs."""
    # Try cyclic groups Z_m
    # For girth >= 10, need no non-backtracking closed walks of length 2-9
    
    for m in range(243, 320):
        if time.time() - start_time > _DEADLINE - 5:
            return None
        
        # Try generating sets {a, b, -a, -b}
        for a in range(1, m // 2 + 1):
            if a == m - a:
                continue
            for b in range(a + 1, m // 2 + 1):
                if b == m - b:
                    continue
                if a == m - b or b == m - a:
                    continue
                
                # Check connectedness
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                # Check girth
                if _check_girth_cyclic(m, a, b, 10):
                    # Construct edges
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    return m, sorted(edge_set)
    
    # Try Z_a x Z_b
    for a in range(2, 20):
        for b in range(a, 20):
            n = a * b
            if n < 243 or n > 320:
                continue
            if time.time() - start_time > _DEADLINE - 5:
                return None
            
            # Try generating sets {(x1,y1), (x2,y2), (-x1,-y1), (-x2,-y2)}
            # For simplicity, try small values
            found = False
            for x1 in range(1, a):
                for y1 in range(1, b):
                    for x2 in range(1, a):
                        for y2 in range(1, b):
                            s1 = (x1, y1)
                            s2 = (x2, y2)
                            ns1 = ((-x1) % a, (-y1) % b)
                            ns2 = ((-x2) % a, (-y2) % b)
                            
                            # Check no degeneracy
                            if s1 == s2 or s1 == ns2 or s2 == ns1:
                                continue
                            
                            # Check girth
                            if _check_girth_abelian(a, b, s1, s2, 10):
                                # Construct edges
                                edge_set = set()
                                for i in range(a):
                                    for j in range(b):
                                        v = (i, j)
                                        for s in [s1, s2]:
                                            nv = ((i + s[0]) % a, (j + s[1]) % b)
                                            idx_v = i * b + j
                                            idx_nv = nv[0] * b + nv[1]
                                            if idx_v < idx_nv:
                                                edge_set.add((idx_v, idx_nv))
                                            else:
                                                edge_set.add((idx_nv, idx_v))
                                return n, sorted(edge_set)
    
    return None


def _check_girth_cyclic(m, a, b, g):
    """Check that Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g."""
    S = [a, -a, b, -b]
    
    for L in range(2, g):
        # Enumerate non-backtracking sequences of length L
        def _enumerate(start, length, current_sum, prev):
            if length == 0:
                return current_sum % m == 0
            for s in S:
                if s == -prev:
                    continue
                if _enumerate(s, length - 1, current_sum + s, s):
                    return True
            return False
        
        for s in S:
            if _enumerate(s, L - 1, s, s):
                return False
    
    return True


def _check_girth_abelian(a, b, s1, s2, g):
    """Check that Cayley graph of Z_a x Z_b with S={s1,s2,-s1,-s2} has girth >= g."""
    S = [s1, s2, ((-s1[0]) % a, (-s1[1]) % b), ((-s2[0]) % a, (-s2[1]) % b)]
    
    for L in range(2, g):
        def _enumerate(start, length, current, prev):
            if length == 0:
                return current == (0, 0)
            for s in S:
                if s == ((-prev[0]) % a, (-prev[1]) % b):
                    continue
                new_current = ((current[0] + s[0]) % a, (current[1] + s[1]) % b)
                if _enumerate(s, length - 1, new_current, s):
                    return True
            return False
        
        for s in S:
            if _enumerate(s, L - 1, s, s):
                return False
    
    return True


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction."""
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
