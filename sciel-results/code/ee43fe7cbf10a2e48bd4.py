import time
import random

def _index(i, j, n):
    """Lexicographic index of pair (i,j) with i<j in the flat list."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color. adj[c][v] is a bitset of neighbors of v in color c."""
    adj = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _index(i, j, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj

def _count_c4s(adj, n):
    """Count total number of C4 violations across all colors."""
    count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    count += 1
    return count

def _is_valid(colors, n):
    """Check if coloring is valid (no monochromatic C4)."""
    adj = _build_adj(colors, n)
    return _count_c4s(adj, n) == 0

def _construct_n25():
    """
    Construct a 5-coloring of K_25 using GF(5) algebraic structure.
    Vertices are (x, y) in GF(5) x GF(5).
    Color is determined by the difference (dx, dy).
    """
    n = 25
    colors = [0] * (n * (n - 1) // 2)
    
    # Precompute field inverses and properties for GF(5)
    # GF(5) = {0, 1, 2, 3, 4}
    # Inverses: 1->1, 2->3, 3->2, 4->4
    
    def get_color(dx, dy):
        # dx, dy are in GF(5)
        if dx == 0 and dy == 0:
            return -1 # Should not happen for distinct vertices
        if dy == 0:
            # Horizontal difference
            # dx is in {1, 2, 3, 4}
            # Map to color 0
            return 0
        if dx == 0:
            # Vertical difference
            # dy is in {1, 2, 3, 4}
            # Map to color 1
            return 1
        
        # Slope m = dx * dy^{-1}
        # dy_inv: 1->1, 2->3, 3->2, 4->4
        inv_dy = {1: 1, 2: 3, 3: 2, 4: 4}[dy]
        m = (dx * inv_dy) % 5
        
        if m == 1:
            return 2
        elif m == 2:
            return 3
        else: # m in {3, 4}
            return 4

    for i in range(n):
        x1, y1 = divmod(i, 5)
        for j in range(i + 1, n):
            x2, y2 = divmod(j, 5)
            dx = (x1 - x2) % 5
            dy = (y1 - y2) % 5
            c = get_color(dx, dy)
            idx = _index(i, j, n)
            colors[idx] = c
            
    # Verify validity
    if _is_valid(colors, n):
        return n, colors
    else:
        return 0, []

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Try the algebraic construction for n=25
    n25, colors25 = _construct_n25()
    if n25 > 0:
        return {'n': n25, 'colors': colors25}
        
    # Fallback: Local search for smaller n
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
            
        num_edges = target_n * (target_n - 1) // 2
        found = False
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = _build_adj(colors, target_n)
            current_violations = _count_c4s(adj, target_n)
            
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                if current_violations == 0:
                    found = True
                    break
                    
                # Simple local search step
                # Find a violation
                viol_found = False
                for c in range(5):
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            common = adj[c][u] & adj[c][v]
                            if common & (common - 1):
                                # Found violation (c, u, v)
                                # Try to fix by recoloring an edge incident to u or v
                                # Pick a random edge from u or v
                                edge_u = random.choice([w for w in range(target_n) if w != u])
                                a, b = sorted((u, edge_u))
                                idx = _index(a, b, target_n)
                                old_c = colors[idx]
                                new_c = random.randint(0, 4)
                                if new_c != old_c:
                                    adj[old_c][a] &= ~(1 << b)
                                    adj[old_c][b] &= ~(1 << a)
                                    adj[new_c][a] |= (1 << b)
                                    adj[new_c][b] |= (1 << a)
                                    colors[idx] = new_c
                                    current_violations = _count_c4s(adj, target_n)
                                viol_found = True
                                break
                        if viol_found: break
                    if viol_found: break
                if not viol_found:
                    found = True
                    break
                    
            if found:
                if _is_valid(colors, target_n):
                    best_n = target_n
                    best_colors = colors[:]
                    break
                    
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}
