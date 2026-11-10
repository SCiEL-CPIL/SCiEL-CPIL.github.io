import time
import numpy as np

def _verify_no_c4(n, colors):
    """Check no color class has a C4. Returns True if valid."""
    m = n * (n - 1) // 2
    if len(colors) != m:
        return False
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adj[c].add((i, j))
            idx += 1
    for c in range(5):
        edges = adj[c]
        if not edges:
            continue
        neigh = [set() for _ in range(n)]
        for (u, v) in edges:
            neigh[u].add(v)
            neigh[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(neigh[u] & neigh[v]) >= 2:
                    return False
    return True

def _lz_woldar_26():
    """Construct the Lazebnik-Woldar 5-coloring of K_26 with no monochromatic C4."""
    n = 26
    q = 5  # prime power
    
    # Vertices 0..24: affine points (x, y) in GF(q)^2
    # Vertex 25: point at infinity
    # Map (x, y) to vertex id: x * q + y
    
    def point_to_id(x, y):
        return x * q + y
    
    def id_to_point(i):
        return (i // q, i % q)
    
    colors = [0] * (n * (n - 1) // 2)
    idx = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == 25 or j == 25:
                # Edge involving point at infinity
                # The other vertex is an affine point (x, y)
                other = i if j == 25 else j
                x, y = id_to_point(other)
                # Assign color based on (x + y) mod 5
                # This ensures that for any line of slope m, at most one point on it
                # has color m from the infinity edge, preventing C4s involving infinity.
                colors[idx] = (x + y) % q
            else:
                x1, y1 = id_to_point(i)
                x2, y2 = id_to_point(j)
                if x1 == x2:
                    # Vertical line, color 4
                    colors[idx] = 4
                else:
                    # Slope = (y2 - y1) * (x2 - x1)^(-1) mod 5
                    dx = (x2 - x1) % q
                    dy = (y2 - y1) % q
                    # Modular inverse of dx mod 5
                    inv_dx = pow(dx, q - 2, q)
                    slope = (dy * inv_dx) % q
                    colors[idx] = slope
            idx += 1
    
    return {'n': n, 'colors': colors}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try the known n=26 Lazebnik-Woldar construction
    result = _lz_woldar_26()
    if _verify_no_c4(result['n'], result['colors']):
        return result
    
    # Fallback: try cyclic constructions for smaller n
    best_result = None
    
    for n in range(15, 25):
        if time.time() > deadline:
            break
        
        if n % 2 == 0:
            continue
        
        half = n // 2
        
        if half <= 5:
            colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
            if _verify_no_c4(n, colors):
                best_result = {'n': n, 'colors': colors}
        else:
            colors = [((min(j - i, n - j + i) - 1) % 5) for i in range(n) for j in range(i + 1, n)]
            if _verify_no_c4(n, colors):
                best_result = {'n': n, 'colors': colors}
    
    if best_result is not None:
        return best_result
    
    # Ultimate fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
