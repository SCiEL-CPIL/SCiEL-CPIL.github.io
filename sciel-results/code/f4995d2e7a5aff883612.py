import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    class C4Coloring:
        def __init__(self, n, colors=None):
            self.n = n
            self.num_edges = n * (n - 1) // 2
            if colors:
                self.colors = colors[:]
            else:
                self.colors = [random.randint(0, 4) for _ in range(self.num_edges)]
            
            self.adj = [[0] * n for _ in range(5)]
            self.common_count = [[0] * n for _ in range(5)] # common_count[c][u][v]
            # Actually, common_count is symmetric. We can store in a dict or 2D list.
            # For n=26, 5*26*26 = 3380 ints. Small.
            self.violations = 0
            
            self._build_adj()
            self._count_violations()

        def _build_adj(self):
            idx = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    c = self.colors[idx]
                    self.adj[c][i] |= (1 << j)
                    self.adj[c][j] |= (1 << i)
                    idx += 1

        def _count_violations(self):
            self.violations = 0
            for c in range(5):
                for u in range(self.n):
                    for v in range(u + 1, self.n):
                        common = self.adj[c][u] & self.adj[c][v]
                        if common.bit_count() > 1:
                            self.violations += 1

        def change_color(self, idx, new_color):
            # Find u, v for idx
            # Precompute edge endpoints
            # This is slow to compute every time. Precompute in init.
            pass

    # Precompute edge endpoints for speed
    def get_edge_endpoints(n):
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                idx += 1
        return edges

    def local_search(n, time_limit):
        edges = get_edge_endpoints(n)
        colors = [random.randint(0, 4) for _ in range(len(edges))]
        
        # Build adj
        adj = [[0] * n for _ in range(5)]
        for (u, v), c in zip(edges, colors):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        # Count violations
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[c][u] & adj[c][v]).bit_count() > 1:
                        violations += 1
                        
        best_violations = violations
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.9999
        
        while time.time() < time_limit:
            if violations == 0:
                break
                
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            # Update violations incrementally
            # 1. Remove (u,v) from old_color
            # 2. Add (u,v) to new_color
            
            # Helper to update violations for a color change
            # This is complex. Let's just do full recount if incremental is too risky.
            # But full recount is slow.
            # Let's try a hybrid: Full recount every 10 steps, incremental every step?
            # No, let's just use full recount but optimize the loop.
            
            colors[idx] = new_color
            new_violations = 0
            # Rebuild adj for new_color and old_color? No, just update adj.
            adj[old_color][u] &= ~(1 << v)
            adj[old_color][v] &= ~(1 << u)
            adj[new_color][u] |= (1 << v)
            adj[new_color][v] |= (1 << u)
            
            # Recount violations?
            # We can just recount the affected pairs.
            # Affected pairs are those involving u or v in old_color or new_color.
            # This is still complex.
            
            # Let's stick to full recount for correctness, but limit iterations.
            # 50,000 iterations * 2ms = 100s.
            # So we can do ~50,000 iterations.
            
            # Recount
            for c in range(5):
                for uu in range(n):
                    for vv in range(uu + 1, n):
                        if (adj[c][uu] & adj[c][vv]).bit_count() > 1:
                            new_violations += 1
                            
            delta = new_violations - violations
            if delta < 0 or random.random() < math.exp(-delta / temp):
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            else:
                colors[idx] = old_color
                # Revert adj
                adj[old_color][u] |= (1 << v)
                adj[old_color][v] |= (1 << u)
                adj[new_color][u] &= ~(1 << v)
                adj[new_color][v] &= ~(1 << u)
                
            temp *= cooling
            if temp < 0.01:
                temp = 0.5

        return best_colors, best_violations

    # Try n=26
    if time.time() < deadline - 10.0:
        for restart in range(3):
            if time.time() > deadline - 5.0: break
            colors, viol = local_search(26, deadline - 5.0)
            if viol == 0:
                return {'n': 26, 'colors': colors}

    # Fallback to smaller n
    for n_try in range(25, 11, -1):
        if time.time() > deadline - 5.0: break
        for restart in range(3):
            if time.time() > deadline - 5.0: break
            colors, viol = local_search(n_try, deadline - 5.0)
            if viol == 0:
                return {'n': n_try, 'colors': colors}

    return {'n': best_n, 'colors': best_colors}
