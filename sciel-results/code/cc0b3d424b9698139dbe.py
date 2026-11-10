# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses algebraic constructions from finite fields (GF(2^m)) to build
    cubic graphs with high girth.
    """
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        return _lnw_construct(k, g)

    best_n = float('inf')
    best_edges = None

    # Strategy 1: Algebraic construction over GF(2^m)
    # We look for a cubic graph on vertices GF(2^m) with girth >= g.
    # The vertices are elements of F = GF(2^m).
    # We define edges based on a set of "steps" or a rational function.
    # A common construction is using a "lifting" or "twisting" of a smaller graph.
    # Alternatively, we can use the construction from "Cubic graphs of girth 16"
    # which often uses a specific polynomial over GF(2^m).
    
    # Let's try a construction based on the group of units or a specific Cayley graph
    # on a finite field extension.
    # For m=10, n=1024.
    # The Moore bound for g=16 is ~512. 1024 is a good target.
    
    # We will try to construct a 3-regular graph on GF(2^m)
    # using a set of 3 involutions (or 1 involution and 1 element of order 2? No, 3 generators).
    # In characteristic 2, involutions are elements of order 2.
    # In a field, the only element of order 2 is 0 (additive) or -1=1 (multiplicative, trivial).
    # So we can't use multiplicative Cayley graphs with involutions directly in GF(2^m)^*.
    # We must use additive Cayley graphs or a different group structure.
    
    # Additive Cayley graph on GF(2^m):
    # Vertices: GF(2^m).
    # Edges: x ~ x + a for a in S, where S is a set of 3 elements.
    # Since char 2, x+a = x-a. So S must be symmetric (it is, if we just pick 3 non-zero elements).
    # But wait, if S = {a, b, c}, the neighbors of x are x+a, x+b, x+c.
    # For the graph to be simple, we need a, b, c distinct and non-zero.
    # Also, we need no short cycles.
    # A cycle of length 2l corresponds to a relation sum_{i=1}^{2l} s_i = 0 where s_i in S, s_i != s_{i+1} (if we consider non-backtracking).
    # Actually, in an additive Cayley graph, a cycle is a sequence s_1, ..., s_k in S such that sum s_i = 0 and no prefix sum is 0 (for simple cycles).
    # We want the minimum length of such a sequence to be >= g.
    
    # This is equivalent to finding a set S of size 3 in GF(2^m) such that the shortest non-trivial linear dependence with coefficients 0 or 1 (and no immediate backtracking? No, just sum=0) has length >= g.
    # Wait, "no cycle shorter than g".
    # In a Cayley graph, the girth is the length of the shortest cycle.
    # A cycle is a sequence of generators s_1, ..., s_k such that s_1 ... s_k = e (in multiplicative notation) or sum s_i = 0 (additive).
    # And the path does not repeat vertices.
    # For additive groups, if sum_{i=1}^k s_i = 0 and k is minimal, and no partial sum is 0, then it's a cycle.
    
    # So we need to find S = {a, b, c} in GF(2^m) such that the smallest k >= g such that there exist s_1, ..., s_k in S with sum s_i = 0 and no partial sum is 0, does not exist for k < g.
    
    # Let's try to find such S for m=10 (n=1024).
    # We can search for S.
    
    m = 10
    n = 1 << m
    
    # We need to represent GF(2^m).
    # Let's use a primitive polynomial.
    # For m=10, x^10 + x^3 + 1 is primitive.
    # We'll implement GF(2^10) arithmetic.
    
    # Precompute exp and log tables for speed?
    # Or just use bit manipulation.
    
    # Let's implement a simple GF(2^m) class or functions.
    # Since m=10, we can use integers 0..1023.
    
    # Primitive polynomial for m=10: 0x0419 (x^10 + x^3 + 1)
    # Let's verify if 0x0419 is primitive for m=10.
    # Common primitive polynomials for m=10:
    # x^10 + x^3 + 1 (0x0419)
    # x^10 + x^7 + x^6 + x^5 + 1 (0x0E01)
    # Let's use 0x0419.
    
    poly = 0x0419
    
    def gf_mul(x, y):
        z = 0
        while y:
            if y & 1:
                z ^= x
            y >>= 1
            x <<= 1
            if x & (1 << m):
                x ^= poly
        return z

    def gf_add(x, y):
        return x ^ y

    def gf_inv(x):
        if x == 0:
            return 0
        # x^(2^m - 2)
        # Or extended Euclidean.
        # For m=10, x^1022.
        # Fast exponentiation.
        res = 1
        base = x
        exp = (1 << m) - 2
        while exp:
            if exp & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            exp >>= 1
        return res

    # We want to find S = {a, b, c} such that the girth is >= g.
    # We can sample random triples (a, b, c) and check the girth.
    # Checking girth:
    # BFS from 0.
    # If we find a cycle of length < g, reject.
    # Since the graph is a Cayley graph, the girth is the length of the shortest relation.
    # We can just do BFS from 0 up to depth g-1. If we don't hit 0 again (except at start), and no other vertex is revisited in a way that forms a shorter cycle?
    # Actually, BFS from 0 finds the shortest path from 0 to any vertex.
    # A cycle of length L exists if there is a vertex v at distance d such that a neighbor of v is at distance d-1 (and not the parent)?
    # No, standard girth calculation:
    # For each vertex, BFS.
    # But for Cayley graph, it's enough to check relations.
    # We can generate all sequences of length < g from S and check if sum is 0.
    # Number of sequences: 3^(g-1). For g=16, 3^15 ~ 14 million. Too slow per trial.
    
    # Better: BFS from 0.
    # Keep track of distance.
    # If we reach a vertex v that was already visited at distance d, and the new path has length d+1, and the parent is not the previous vertex in the path?
    # Actually, just do BFS and look for the shortest cycle.
    # In a Cayley graph, the shortest cycle is the shortest word in S that evaluates to 0.
    # We can do a BFS on the "state space" of the walk? No, that's huge.
    # We can just do a standard BFS on the graph and detect the first back-edge that forms a cycle.
    # Since the graph is undirected, we avoid backtracking to parent.
    
    # Let's try a few random triples.
    
    random.seed(42)
    
    max_trials = 2000
    found = False
    
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
            
        # Pick 3 random non-zero distinct elements
        # Ensure they are linearly independent? Not necessarily, but helps girth.
        a = random.randint(1, n - 1)
        b = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        
        if a == b or b == c or a == c:
            continue
            
        S = [a, b, c]
        
        # Check girth
        # BFS from 0
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = deque([0])
        min_cycle = n + 1
        
        while queue:
            u = queue.popleft()
            if dist[u] >= min_cycle:
                continue
            for s in S:
                v = gf_add(u, s)
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # Check if it's a back edge to parent
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            
        if min_cycle >= g:
            # Found a valid graph
            # Build edges
            edges = []
            for u in range(n):
                for s in S:
                    v = gf_add(u, s)
                    if u < v:
                        edges.append((u, v))
            
            if n < best_n:
                best_n = n
                best_edges = edges
            found = True
            # We can break if we found one, but maybe a smaller m exists?
            # m=10 gives 1024.
            # m=9 gives 512. Moore bound for g=16 is 512.
            # If we find one for m=9, it's better.
            # Let's try m=9 first.
            break

    if not found:
        # Try m=9
        m = 9
        n = 1 << m
        # Primitive polynomial for m=9: x^9 + x^4 + 1 (0x0211)
        poly = 0x0211
        
        def gf_mul(x, y):
            z = 0
            while y:
                if y & 1:
                    z ^= x
                y >>= 1
                x <<= 1
                if x & (1 << m):
                    x ^= poly
            return z

        def gf_add(x, y):
            return x ^ y

        random.seed(43)
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            a = random.randint(1, n - 1)
            b = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            
            if a == b or b == c or a == c:
                continue
                
            S = [a, b, c]
            
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            queue = deque([0])
            min_cycle = n + 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_cycle:
                    continue
                for s in S:
                    v = gf_add(u, s)
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                
            if min_cycle >= g:
                edges = []
                for u in range(n):
                    for s in S:
                        v = gf_add(u, s)
                        if u < v:
                            edges.append((u, v))
                
                if n < best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break

    if best_edges is not None:
        return best_n, best_edges

    # Fallback to LUW Construction
    for d in range(6, 16):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            best_n = n
            best_edges = edges
            return n, edges

    if best_edges is not None:
        return best_n, best_edges
        
    return _lnw_construct(3, g)

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    q = k
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

def _lnw_construct(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
