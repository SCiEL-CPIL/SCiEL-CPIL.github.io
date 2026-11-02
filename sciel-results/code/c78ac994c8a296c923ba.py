import time
import random
from collections import deque

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
    max_n_search = max(moore_bound + 1000, best_n - 100)
    
    # Use a deterministic pseudo-random sequence for reproducibility
    random.seed(42)
    
    found_circulant = False
    current_n = moore_bound
    if current_n % 2 != 0:
        current_n += 1
        
    while current_n < max_n_search:
        if time.time() > deadline:
            break
            
        # Try a few random generators for this n
        # To increase success rate, we can try structured generators too
        # e.g., a=1 is bad (creates small cycles often), so a>=2
        attempts = 20
        for _ in range(attempts):
            if time.time() > deadline:
                break
            a = random.randint(2, current_n // 2 - 2)
            b = random.randint(2, current_n // 2 - 2)
            if a == b:
                continue
            # Ensure distinct
            if a == current_n - b or b == current_n - a: # Same edge
                continue
            if a == current_n // 2 or b == current_n // 2:
                continue
                
            # Construct S
            # S = {a, -a, b, -b, n/2}
            # Check girth
            girth = _check_circulant_girth(current_n, a, b)
            
            if girth >= g:
                # Verify it's simple and degree k
                # It is by construction if a,b,n/2 are distinct and non-zero
                edges = _build_circulant_edges(current_n, a, b)
                if len(edges) == k * current_n // 2: # 5 * n / 2
                    best_n = current_n
                    best_edges = edges
                    found_circulant = True
                    # Since we are iterating n upwards, the first valid n is the smallest
                    # We can break out of the n loop immediately
                    break
        
        if found_circulant:
            break
            
        current_n += 2

    return best_n, best_edges

def _check_circulant_girth(n, a, b):
    """
    Checks the girth of the circulant graph Cay(Z_n, {a, -a, b, -b, n/2}).
    Returns the length of the shortest cycle.
    """
    # BFS on the graph to find shortest cycle
    # Since it's a Cayley graph, we can just BFS from 0
    # But we need to avoid backtracking.
    # State: (current_node, last_step_used)
    # last_step_used is one of {a, -a, b, -b, n/2, None}
    
    steps = [a, n - a, b, n - b, n // 2]
    # Map step value to index for quick lookup?
    # Actually, just storing the step value is fine.
    
    # dist[v][last_step] = distance
    # n is small (< 2000), steps are 5.
    # 2000 * 6 = 12000 states. Very fast.
    
    INF = 10**9
    # dist[v][s] where s is 0..4 representing the step taken to arrive at v
    # If s == 5, it means "no previous step" (start)
    dist = [[INF] * 6 for _ in range(n)]
    dist[0][5] = 0
    
    queue = deque()
    queue.append((0, 5))
    
    min_cycle = INF
    
    while queue:
        v, last_s = queue.popleft()
        d = dist[v][last_s]
        
        # If we are back at 0 with a step, we found a cycle
        if v == 0 and last_s != 5:
            if d < min_cycle:
                min_cycle = d
            # We can continue to find potentially smaller? No, BFS finds shortest first.
            # But we start at 0, so the first time we hit 0 with a step is the shortest cycle through 0.
            # In a vertex-transitive graph, the girth is the shortest cycle through any vertex.
            # So this is sufficient.
            break
            
        for i, step in enumerate(steps):
            next_v = (v + step) % n
            # Avoid backtracking: cannot go back to the node we came from
            # The node we came from was (v - step) % n
            # But we track last_s. The previous node was (v - steps[last_s]) % n if last_s != 5
            # Actually, simpler:
            # If last_s != 5, the step taken to get to v was steps[last_s].
            # The previous node was prev_v = (v - steps[last_s]) % n.
            # We cannot take a step 'j' such that (v + steps[j]) % n == prev_v.
            # This is equivalent to steps[j] == -steps[last_s] (mod n).
            
            if last_s != 5:
                # Check if this step is the inverse of the last step
                # In Z_n, inverse of x is (n-x) % n.
                # steps are a, n-a, b, n-b, n/2.
                # Inverse of a is n-a. Inverse of n-a is a.
                # Inverse of n/2 is n/2.
                
                # We just check if the next node is the previous node.
                prev_v = (v - steps[last_s]) % n
                if next_v == prev_v:
                    continue
            
            if dist[next_v][i] > d + 1:
                dist[next_v][i] = d + 1
                queue.append((next_v, i))
                
    return min_cycle

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
