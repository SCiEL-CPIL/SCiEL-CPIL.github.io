import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    # Petersen Graph as base
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length < g from every vertex
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient vectors for each walk
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        # Store as list of (index, coefficient) for non-zero entries
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_coeffs.append(nonzero)
    
    # Filter to only minimal walks (those that can't be decomposed)
    # Keep all for now, but deduplicate by coefficient vector
    seen_coeffs = set()
    filtered_walk_coeffs = []
    for wc in walk_coeffs:
        key = tuple(wc)
        if key not in seen_coeffs:
            seen_coeffs.add(key)
            filtered_walk_coeffs.append(wc)
    walk_coeffs = filtered_walk_coeffs
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    def check_valid(volts, m):
        """Check if voltage assignment gives girth >= g"""
        for nonzero in walk_coeffs:
            s = 0
            for idx, c in nonzero:
                s += c * volts[idx]
            if s % m == 0:
                return False
        return True
    
    def build_lift(m, volts):
        """Build the lifted graph edges"""
        lift_edges = []
        for i, (u, v) in enumerate(edges):
            vol = volts[i]
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + (t + vol) % m
                lift_edges.append((u_lift, v_lift))
        return lift_edges
    
    # Strategy 1: Systematic polynomial voltage search over primes
    # This explores a structured family that's much larger than random
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Try primes from Moore bound upward
    # n = 10 * p, so p >= 52 means p >= 53 (prime)
    p = 53
    while p < 1000 and time.time() < deadline - 20.0:
        if not is_prime(p):
            p += 1
            continue
        
        candidate_n = 10 * p
        if candidate_n >= best_n:
            p += 1
            continue
        
        # Systematic search: try voltage assignments of the form
        # volts[i] = f(i) mod p where f is a polynomial of degree d
        # Try different polynomial coefficients
        found = False
        
        # For small primes, do more exhaustive search
        if p <= 200:
            # Try all linear assignments: volts[i] = a*i + b mod p
            for a in range(1, p):
                if time.time() > deadline - 10.0:
                    break
                for b in range(p):
                    volts = [(a * i + b) % p for i in range(len(edges))]
                    # Ensure connectivity: gcd of all volts with p should be 1
                    # Since p is prime, this means at least one volt is nonzero
                    if any(v != 0 for v in volts):
                        if check_valid(volts, p):
                            found = True
                            best_m = p
                            best_n = candidate_n
                            best_edges = build_lift(p, volts)
                            break
                if found:
                    break
        else:
            # For larger primes, try quadratic and random
            # Try quadratic: volts[i] = a*i^2 + b*i + c mod p
            for a in range(1, min(p, 50)):
                if time.time() > deadline - 10.0:
                    break
                for b in range(min(p, 50)):
                    for c in range(min(p, 50)):
                        volts = [(a * i * i + b * i + c) % p for i in range(len(edges))]
                        if any(v != 0 for v in volts):
                            if check_valid(volts, p):
                                found = True
                                best_m = p
                                best_n = candidate_n
                                best_edges = build_lift(p, volts)
                                break
                    if found:
                        break
                if found:
                    break
            
            # Also try random for larger primes
            if not found:
                max_tries = min(1000, max(100, int((deadline - time.time() - 10) / 0.0001)))
                for trial in range(max_tries):
                    if time.time() > deadline - 10.0:
                        break
                    volts = [random.randint(1, p - 1) for _ in range(len(edges))]
                    if check_valid(volts, p):
                        found = True
                        best_m = p
                        best_n = candidate_n
                        best_edges = build_lift(p, volts)
                        break
        
        if found:
            break
        p += 1
    
    # Strategy 2: If no prime found, try composite m with random search
    if best_edges is None:
        for m in range(52, 500):
            if time.time() > deadline - 5.0:
                break
            if m % 2 == 0 or m % 5 == 0:  # Skip even and multiples of 5 for efficiency
                continue
            
            candidate_n = 10 * m
            if candidate_n >= best_n:
                continue
            
            time_left = deadline - time.time() - 5.0
            max_tries = min(2000, max(100, int(time_left / 0.0002)))
            
            found = False
            for trial in range(max_tries):
                if time.time() > deadline - 5.0:
                    break
                
                volts = [random.randint(1, m - 1) for _ in range(len(edges))]
                
                # Quick connectivity check
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val != 1:
                    continue
                
                if check_valid(volts, m):
                    found = True
                    best_m = m
                    best_n = candidate_n
                    best_edges = build_lift(m, volts)
                    break
            
            if found:
                break
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    # Deduplicate edges
    edge_set = set()
    unique_edges = []
    for e in edges_luw:
        key = (min(e), max(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)
    return len(order), unique_edges

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
