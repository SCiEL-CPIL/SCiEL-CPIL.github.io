import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 4:
        return 0, []
    
    # Priority 1: Known 320-vertex (4,10)-cage via voltage graph
    if g == 10:
        result = _construct_cage_320()
        if result is not None:
            n, edges = result
            if _verify(n, edges, k, g):
                return n, edges
    
    # Priority 2: Voltage graph search for other girths or if above failed
    if time.time() < deadline - 25:
        result = _voltage_search(k, g, deadline)
        if result is not None:
            n, edges = result
            if _verify(n, edges, k, g):
                return n, edges
    
    # Priority 3: Fallback to LUW
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
    
    return 0, []

def _construct_cage_320():
    """Construct the 320-vertex (4,10)-cage using the known voltage graph.
    
    The base graph is a 16-vertex 4-regular graph (16-cycle with 4 chords).
    Voltage group is Z_16. We search for valid voltages that give girth >= 10.
    """
    m = 16
    
    # Base graph: 16-cycle with 4 chords
    # Cycle: (i, i+1 mod 16) for i=0..15
    # Chords: connect vertices 8 apart at positions 0,1,2,3
    base_edges = []
    for i in range(16):
        base_edges.append((i, (i + 1) % 16))
    for i in range(4):
        base_edges.append((i, (i + 8) % 16))
    
    n_edges = len(base_edges)  # 20 edges
    n = m * 16  # 320 vertices
    
    # Enumerate all closed non-backtracking walks of length < 10 in base graph
    walks = _enumerate_short_walks(base_edges, g=10)
    
    if not walks:
        # No short walks to avoid - any voltages work
        return n, _build_lift(base_edges, [0]*n_edges, m)
    
    # Build constraint system: for each walk, sum(voltage[e]*sign) != 0 mod m
    walk_coeffs = []
    for walk in walks:
        coeffs = {}
        for eidx, sign in walk:
            coeffs[eidx] = coeffs.get(eidx, 0) + sign
        walk_coeffs.append(coeffs)
    
    # Try to find voltages that avoid all zero-sum walks
    # Use the known structure: cycle edges get voltage 1, chords get specific values
    # Try systematic search with good initial guesses
    
    best_voltages = None
    
    # Strategy: cycle edges mostly get small voltages, chords get larger
    # Known good pattern: cycle=1, chords vary
    
    # Try a structured search
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
            
        if attempt == 0:
            # Known pattern: all cycle edges voltage 1, chords voltage 8
            vols = [1]*16 + [8, 8, 8, 8]
        elif attempt == 1:
            # Cycle voltage 1, chords voltage 4
            vols = [1]*16 + [4, 4, 4, 4]
        elif attempt == 2:
            # Cycle voltage 1, chords voltage 12
            vols = [1]*16 + [12, 12, 12, 12]
        elif attempt == 3:
            # Cycle voltage 3, chords voltage 8
            vols = [3]*16 + [8, 8, 8, 8]
        elif attempt == 4:
            # Cycle voltage 5, chords voltage 8
            vols = [5]*16 + [8, 8, 8, 8]
        elif attempt == 5:
            # Cycle voltage 7, chords voltage 8
            vols = [7]*16 + [8, 8, 8, 8]
        elif attempt == 6:
            # Cycle voltage 1, chords voltage 2
            vols = [1]*16 + [2, 2, 2, 2]
        elif attempt == 7:
            # Cycle voltage 1, chords voltage 6
            vols = [1]*16 + [6, 6, 6, 6]
        elif attempt == 8:
            # Cycle voltage 1, chords voltage 10
            vols = [1]*16 + [10, 10, 10, 10]
        elif attempt == 9:
            # Cycle voltage 1, chords voltage 14
            vols = [1]*16 + [14, 14, 14, 14]
        else:
            # Random with bias toward small cycle voltages
            vols = [random.choice([1, 3, 5, 7, 9, 11, 13, 15]) for _ in range(16)]
            vols += [random.randint(1, 15) for _ in range(4)]
        
        if _check_voltages_valid(vols, walk_coeffs, m):
            best_voltages = vols
            break
        
        # Local search from this starting point
        if attempt < 20:
            improved = _local_search_voltages(vols, walk_coeffs, m, deadline - 15)
            if improved is not None:
                best_voltages = improved
                break
    
    if best_voltages is None:
        return None
    
    lift_edges = _build_lift(base_edges, best_voltages, m)
    return n, lift_edges

def _enumerate_short_walks(base_edges, g):
    """Enumerate all closed non-backtracking walks of length < g in base graph."""
    base_n = len(set(v for e in base_edges for v in e))
    adj = [[] for _ in range(base_n)]
    edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)
    
    walks = []
    
    for L in range(3, g):
        for u, v in base_edges:
            idx, sign = edge_map[(u, v)]
            path = [(idx, sign)]
            prev = u
            cur = v
            
            def dfs(prev, cur, path, target_u, target_L):
                if len(path) == target_L:
                    if cur == target_u:
                        walks.append(path[:])
                    return
                for (next_v, eidx) in adj[cur]:
                    if next_v == prev:
                        continue
                    e_u, e_v = base_edges[eidx]
                    if e_u == cur and e_v == next_v:
                        s = 1
                    else:
                        s = -1
                    dfs(cur, next_v, path + [(eidx, s)], target_u, target_L)
            
            dfs(prev, cur, path, u, L)
    
    return walks

def _check_voltages_valid(vols, walk_coeffs, m):
    """Check if voltages avoid all zero-sum walks."""
    for coeffs in walk_coeffs:
        s = 0
        for eidx, coeff in coeffs.items():
            s = (s + coeff * vols[eidx]) % m
        if s == 0:
            return False
    return True

def _local_search_voltages(vols, walk_coeffs, m, deadline):
    """Local search to find valid voltages."""
    current = vols[:]
    n_edges = len(current)
    
    def count_violations(v):
        count = 0
        for coeffs in walk_coeffs:
            s = 0
            for eidx, coeff in coeffs.items():
                s = (s + coeff * v[eidx]) % m
            if s == 0:
                count += 1
        return count
    
    current_violations = count_violations(current)
    if current_violations == 0:
        return current
    
    T = 3.0
    T_min = 0.01
    cooling = 0.95
    steps = 0
    max_steps = 1000
    
    while T > T_min and steps < max_steps and time.time() < deadline:
        e_idx = random.randint(0, n_edges - 1)
        old_v = current[e_idx]
        new_v = random.randint(0, m - 1)
        if new_v == old_v:
            continue
        
        current[e_idx] = new_v
        new_violations = count_violations(current)
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_violations = new_violations
        else:
            current[e_idx] = old_v
        
        T *= cooling
        steps += 1
        
        if current_violations == 0:
            return current[:]
    
    return None

def _build_lift(base_edges, voltages, m):
    """Build the voltage lift graph."""
    base_n = len(set(v for e in base_edges for v in e))
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
                
    return sorted(set(edges))

def _voltage_search(k, g, deadline):
    """General voltage graph search for other girths."""
    if k != 4:
        return None
        
    base_graphs = [
        ("K44", _get_k44_edges()),
    ]
    
    best_result = None
    best_n = float('inf')
    
    for base_name, base_edges in base_graphs:
        base_n = len(set(v for e in base_edges for v in e))
        max_m = min(40, int(300 / base_n) + 5)
        
        for m in range(10, max_m):
            if time.time() > deadline - 10:
                break
                
            n = m * base_n
            if n >= best_n:
                continue
                
            voltages = _find_voltages(base_edges, m, g, deadline - 5)
            if voltages is not None:
                lift_edges = _build_lift(base_edges, voltages, m)
                if _verify(n, lift_edges, k, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, lift_edges)
                        
    return best_result

def _get_k44_edges():
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges

def _find_voltages(base_edges, m, g, deadline):
    """Find valid voltages for a base graph."""
    n_edges = len(base_edges)
    walks = _enumerate_short_walks(base_edges, g)
    
    if not walks:
        return [0] * n_edges
        
    walk_coeffs = []
    for walk in walks:
        coeffs = {}
        for eidx, sign in walk:
            coeffs[eidx] = coeffs.get(eidx, 0) + sign
        walk_coeffs.append(coeffs)
    
    random.seed(999)
    
    for restart in range(5):
        if time.time() > deadline - 2:
            break
        vols = [random.randint(0, m - 1) for _ in range(n_edges)]
        
        if _check_voltages_valid(vols, walk_coeffs, m):
            return vols[:]
        
        improved = _local_search_voltages(vols, walk_coeffs, m, deadline - 1)
        if improved is not None:
            return improved
    
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    return n, edges_list

def _verify(n, edges, k, g):
    """Verify the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1 or len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Connectivity check
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check using BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            if dist[v] >= g - 1:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                elif dist[w] >= dist[v]:
                    if dist[v] + dist[w] + 1 < g:
                        return False
    return True
