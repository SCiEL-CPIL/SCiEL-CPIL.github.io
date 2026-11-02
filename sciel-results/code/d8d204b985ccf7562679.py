import time
import random
from collections import deque
from math import gcd

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses cyclic Cayley graphs Z_N with generators {a, -a, b}.
    Searches for the smallest N and generators that satisfy the girth constraint.
    """
    if k != 3:
        # Fallback to seed logic for other k
        return _luw_fallback(k, g)

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def check_girth_circulant(N, a, b, target_girth):
        """
        Checks if the circulant graph Z_N with generators a, b has girth >= target_girth.
        Uses BFS on the group (sum, last_gen) to find the shortest non-backtracking cycle.
        Returns True if girth >= target_girth.
        """
        # Generators: a, -a, b, -b
        # Indices: 0: a, 1: -a, 2: b, 3: -b
        # Inverses: 0<->1, 2<->3
        
        # dist[sum][last_gen] = length
        # Initialize with -1
        dist = [[-1 for _ in range(4)] for _ in range(N)]
        
        queue = deque()
        # Start at 0, no last gen (represented by -1, but we handle start separately)
        # We start by taking a step.
        # Actually, we can start with (0, -1) and length 0.
        
        # To avoid special casing, we push the neighbors of 0.
        # But we need to track last_gen to prevent backtracking.
        
        # Let's push (0, -1, 0)
        queue.append((0, -1, 0))
        
        while queue:
            s, last_gen, length = queue.popleft()
            
            if length >= target_girth:
                continue
                
            gens = [a, -a, b, -b]
            
            for i in range(4):
                gen = gens[i]
                
                # Check non-backtracking
                if last_gen != -1:
                    if (last_gen == 0 and i == 1) or (last_gen == 1 and i == 0) or \
                       (last_gen == 2 and i == 3) or (last_gen == 3 and i == 2):
                        continue
                
                new_s = (s + gen) % N
                
                # If we return to 0, we have a cycle
                if new_s == 0 and length + 1 >= 3:
                    if length + 1 < target_girth:
                        return False
                    else:
                        # Found a cycle of sufficient length.
                        # Since BFS, this is the shortest.
                        return True
                
                if dist[new_s][i] == -1:
                    dist[new_s][i] = length + 1
                    queue.append((new_s, i, length + 1))
                    
        return True

    def generate_cyclic_graph(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid.
        """
        a = a % N
        b = b % N
        
        # Check for simple graph conditions
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N: # 2a = 0
            return None
        if b == (N - b) % N: # 2b = 0
            return None
        if a == b:
            return None
        if a == (N - b) % N:
            return None
        if (N - a) % N == b:
            return None
        if (N - a) % N == (N - b) % N:
            return None

        # Check connectivity: gcd(a, b, N) must be 1
        if gcd(gcd(a, b), N) != 1:
            return None

        edges = set()
        for i in range(N):
            j1 = (i + a) % N
            j2 = (i - a) % N
            j3 = (i + b) % N
            
            for j in [j1, j2, j3]:
                if i > j:
                    u, v = j, i
                else:
                    u, v = i, j
                if u == v:
                    return None
                edges.add((u, v))
            
        return N, list(edges)

    random.seed(42)
    
    # Search N from Moore bound upwards
    # Moore bound for k=3, g=16 is 512.
    # Known best is 936.
    # We search up to 1000 to find something better than LUW (1280).
    
    max_N = 1000
    
    for N in range(512, max_N + 1):
        if time.time() > deadline - 5:
            break
            
        attempts = 0
        # Increase attempts for smaller N?
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            a = random.randint(1, N - 1)
            b = random.randint(1, N - 1)
            
            # Fast pre-check
            if a % N == 0 or b % N == 0: continue
            if a == (N - a) % N or b == (N - b) % N: continue
            if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N: continue
            
            # Connectivity check
            if gcd(gcd(a, b), N) != 1: continue
            
            # Check girth
            if check_girth_circulant(N, a, b, g):
                res = generate_cyclic_graph(N, a, b)
                if res is not None:
                    return res

    # Fallback to LUW if no circulant found
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction fallback.
    """
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
