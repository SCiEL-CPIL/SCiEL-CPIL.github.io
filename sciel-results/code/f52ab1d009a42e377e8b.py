import itertools
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Try the specific group Z_2 x Z_2 x Z_5 x Z_16 for g=10
    if g == 10:
        result = _try_cayley_z2z2z5z16()
        if result:
            return result
    
    # Fallback to previous logic if specific group fails or g != 10
    # (Though the task is specific to k=4, g=10 usually)
    return _construct_fallback(k, g)

def _try_cayley_z2z2z5z16():
    # Group G = Z_2 x Z_2 x Z_5 x Z_16
    # Elements are tuples (a, b, c, d)
    # Identity is (0, 0, 0, 0)
    
    # Pre-compute exponent pairs for non-backtracking words of length 2 to 9
    # Symbols: 1 (s1), -1 (s1^-1), 2 (s2), -2 (s2^-1)
    # We map s1 to vector (1, 0) in exponent space, s2 to (0, 1).
    # A word is a sequence of these.
    # Non-backtracking: consecutive symbols cannot be inverses.
    # Inverse of 1 is -1. Inverse of 2 is -2.
    
    exponents = set()
    
    def add_word(word, k1, k2):
        if len(word) >= 2:
            exponents.add((k1, k2))
            
    # Recursive generation of words
    def generate(prefix, k1, k2, length):
        if length == 0:
            if len(prefix) >= 2:
                exponents.add((k1, k2))
            return
        
        last = prefix[-1] if prefix else None
        for sym in [1, -1, 2, -2]:
            # Check non-backtracking
            if last is not None:
                if sym == -last:
                    continue
            # Also, we don't want to generate words that are "trivial" in a way?
            # No, just generate all non-backtracking words.
            
            new_k1 = k1 + (1 if sym == 1 else -1 if sym == -1 else 0)
            new_k2 = k2 + (1 if sym == 2 else -1 if sym == -2 else 0)
            
            generate(prefix + [sym], new_k1, new_k2, length - 1)
            
    # We need words of length 2 to 9
    for L in range(2, 10):
        # Start with each symbol
        for sym in [1, -1, 2, -2]:
            k1 = 1 if sym == 1 else -1 if sym == -1 else 0
            k2 = 1 if sym == 2 else -1 if sym == -2 else 0
            generate([sym], k1, k2, L - 1)
            
    # Convert exponents to a list for faster checking
    exp_list = list(exponents)
    
    # Define the group elements
    # a in 0..1, b in 0..1, c in 0..4, d in 0..15
    # Total 2*2*5*16 = 320 elements
    elements = [(a, b, c, d) for a in range(2) for b in range(2) for c in range(5) for d in range(16)]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Group operation
    def add_g(e1, e2):
        return ((e1[0]+e2[0])%2, (e1[1]+e2[1])%2, (e1[2]+e2[2])%5, (e1[3]+e2[3])%16)
        
    def neg_g(e):
        return ((-e[0])%2, (-e[1])%2, (-e[2])%5, (-e[3])%16)
        
    def mul_g(k, e):
        if k == 0: return (0,0,0,0)
        if k == 1: return e
        if k == -1: return neg_g(e)
        # For larger k, use repeated addition or fast exponentiation
        # But k is small (from words of length < 10, max |k| is 9)
        if k > 0:
            res = (0,0,0,0)
            for _ in range(k):
                res = add_g(res, e)
            return res
        else:
            return neg_g(mul_g(-k, e))
            
    # Iterate over pairs s1, s2
    # s1, s2 must be non-identity
    # s1 != s2, s1 != -s2, etc. to ensure degree 4 and no multiple edges
    
    for s1 in elements:
        if s1 == (0,0,0,0): continue
        for s2 in elements:
            if s2 == (0,0,0,0): continue
            if s1 == s2: continue
            if s1 == neg_g(s2): continue
            if s2 == neg_g(s1): continue # Same as above
            
            # Check girth condition
            valid = True
            for k1, k2 in exp_list:
                # Compute k1*s1 + k2*s2
                term1 = mul_g(k1, s1)
                term2 = mul_g(k2, s2)
                res = add_g(term1, term2)
                if res == (0,0,0,0):
                    valid = False
                    break
            
            if valid:
                # Found a valid generating set
                # Construct edges
                edges = set()
                for i, v in enumerate(elements):
                    # Neighbors: v+s1, v-s1, v+s2, v-s2
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

def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
