import time
import random
from collections import deque
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Tries to find a small circulant graph first, falls back to LUW.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Calculate a rough Moore bound for k=5, g=9
    # Moore bound M(k,g) for girth g (odd): 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if k == 5 and g == 9:
        moore_bound = 428
    else:
        # General estimation
        if g % 2 == 1:
            r = (g - 3) // 2
        else:
            r = (g - 2) // 2
        moore_bound = 1 + k * sum([(k - 1) ** i for i in range(r + 1)])
        
    # Fallback: LUW Construction (always valid)
    luw_result = _luw_construct(k, g)
    best_n, best_edges = luw_result
    
    # 2. Search for Circulant Graphs
    # A 5-regular circulant graph on Z_n requires n to be even.
    # Generators: {a, n-a, b, n-b, n/2}
    # We search for n >= moore_bound
    max_n_search = max(moore_bound + 2000, best_n - 50)
    
    # Use a deterministic pseudo-random sequence for reproducibility
    random.seed(42)
    
    found_circulant = False
    current_n = moore_bound
    if current_n % 2 != 0:
        current_n += 1
        
    while current_n < max_n_search:
        if time.time() > deadline:
            break
            
        # Try structured generators first, then random
        # Structured: a small, b related to n
        structured_pairs = []
        half = current_n // 2
        # Small a values
        for a in range(2, min(10, half - 2)):
            for b in range(a + 1, min(a + 50, half - 1)):
                if b != half and a != b and a + b != current_n and abs(a - b) != half:
                    structured_pairs.append((a, b))
        # Also try a near sqrt(n), b near n/sqrt(n)
        sqrt_n = int(math.sqrt(current_n))
        for a in range(max(2, sqrt_n - 5), min(sqrt_n + 5, half - 2)):
            for b in range(max(a + 1, int(current_n / (a + 1)) - 5), min(int(current_n / (a + 1)) + 5, half - 1)):
                if b != half and a != b and a + b != current_n and abs(a - b) != half:
                    if (a, b) not in structured_pairs:
                        structured_pairs.append((a, b))
        
        # Try structured pairs first
        for a, b in structured_pairs[:30]:
            if time.time() > deadline:
                break
            girth = _check_circulant_girth(current_n, a, b)
            if girth >= g:
                edges = _build_circulant_edges(current_n, a, b)
                if len(edges) == k * current_n // 2:
                    best_n = current_n
                    best_edges = edges
                    found_circulant = True
                    break
        
        if found_circulant:
            break
        
        # Then try random pairs
        attempts = 15
        for _ in range(attempts):
            if time.time() > deadline:
                break
            a = random.randint(2, half - 2)
            b = random.randint(2, half - 2)
            if a == b or a == half or b == half:
                continue
            if a + b == current_n or abs(a - b) == half:
                continue
            girth = _check_circulant_girth(current_n, a, b)
            if girth >= g:
                edges = _build_circulant_edges(current_n, a, b)
                if len(edges) == k * current_n // 2:
                    best_n = current_n
                    best_edges = edges
                    found_circulant = True
                    break
        
        if found_circulant:
            break
            
        current_n += 2

    return best_n, best_edges

def _check_circulant_girth(n, a, b):
    """
    Checks the girth of the circulant graph Cay(Z_n, {a, -a, b, -b, n/2}).
    Returns the length of the shortest cycle.
    Uses non-backtracking BFS from vertex 0.
    """
    steps = [a, n - a, b, n - b, n // 2]
    
    INF = 10**9
    # dist[v][s] where s is 0..4 representing the step taken to arrive at v
    # If s == 5, it means "no previous step" (start)
    dist = [[INF] * 6 for _ in range(n)]
    dist[0][5] = 0
    
    queue = deque()
    queue.append((0, 5))
    
    # Precompute inverse step indices for fast backtracking check
    # inv_step[i] = j such that steps[i] + steps[j] == 0 (mod n)
    inv_step = [0] * 5
    for i in range(5):
        for j in range(5):
            if (steps[i] + steps[j]) % n == 0:
                inv_step[i] = j
                break
    
    while queue:
        v, last_s = queue.popleft()
        d = dist[v][last_s]
        
        # If we are back at 0 with a step, we found a cycle
        if v == 0 and last_s != 5:
            return d
            
        for i in range(5):
            # Skip the inverse of the last step (backtracking)
            if last_s != 5 and i == inv_step[last_s]:
                continue
            
            next_v = (v + steps[i]) % n
            
            if dist[next_v][i] > d + 1:
                dist[next_v][i] = d + 1
                queue.append((next_v, i))
                
    return INF

def _build_circulant_edges(n, a, b):
    steps = [a, n - a, b, n - b, n // 2]
    edges = []
    seen = set()
    for u in range(n):
        for v_step in steps:
            v = (u + v_step) % n
            if u < v:
                edge = (u, v)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
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
# EVOLVE-BLOCK-END
