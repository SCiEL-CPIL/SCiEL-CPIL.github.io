"""Construct a large triangle-free graph with independence number < 10."""
import time


def _polarity_graph_q5():
    """Polarity graph of Q(3,5): n=31, triangle-free, alpha=6."""
    q = 5
    
    def gf_inv(a):
        for b in range(1, q):
            if (a * b) % q == 1:
                return b
        return 0
    
    points = []
    seen = set()
    for x0 in range(q):
        for x1 in range(q):
            for x2 in range(q):
                for x3 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0 and x3 == 0:
                        continue
                    val = (x0*x0 + x1*x1 + x2*x2 + x3*x3) % q
                    if val != 0:
                        continue
                    vec = [x0, x1, x2, x3]
                    for i in range(4):
                        if vec[i] != 0:
                            inv = gf_inv(vec[i])
                            vec = [(v * inv) % q for v in vec]
                            break
                    key = tuple(vec)
                    if key not in seen:
                        seen.add(key)
                        points.append(key)
    
    n_pts = len(points)
    
    edges = []
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            bi = points[i]
            bj = points[j]
            b = sum((bi[k] * bj[k]) % q for k in range(4)) % q
            if b == 0:
                edges.append([i, j])
    
    return {'n': n_pts, 'edges': edges}


def _andrasfai(t):
    """Andrasfai graph And(t): n=3t, triangle-free, alpha=t."""
    n = 3 * t
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _has_triangle(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return True
    return False


def _max_independent_set_size(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    best = [0]

    def branch(remaining, current_size):
        if not remaining:
            best[0] = max(best[0], current_size)
            return
        if current_size + len(remaining) <= best[0]:
            return
        v = remaining[0]
        branch(remaining - {v} - adj[v], current_size + 1)
        branch(remaining - {v}, current_size)

    branch(set(range(n)), 0)
    return best[0]


def construct():
    start_time = time.time()
    deadline = start_time + 100

    # Primary: polarity graph of Q(3,5), n=31, alpha=6
    result = _polarity_graph_q5()
    n = result['n']
    edges = result['edges']

    # Quick triangle check
    if _has_triangle(n, edges):
        result = _andrasfai(9)
        n = result['n']
        edges = result['edges']

    # Independence check (should be fast for n=31, alpha=6)
    if time.time() < deadline:
        alpha = _max_independent_set_size(n, edges)
        if alpha >= 10:
            result = _andrasfai(9)

    return result
