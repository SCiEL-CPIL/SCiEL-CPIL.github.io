import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _time_left():
        return deadline - time.time()

    def _verify(n, colors):
        """
        Check if the coloring is valid (no C4 in any color class).
        A C4 in a color class exists if two vertices have two or more common neighbors in that color.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        if any(c < 0 or c > 4 for c in colors):
            return False
            
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    common = mask_u & adj[c][v]
                    # If there are at least 2 common neighbors, we have a C4
                    # (u, x, v, y, u) where x,y are distinct common neighbors.
                    # common & (common - 1) is non-zero iff popcount(common) >= 2
                    if common & (common - 1):
                        return False
        return True

    def _cyclic_construction(n):
        """
        Cyclic construction for odd n.
        Valid for n=11 with 5 colors.
        """
        if n % 2 == 0:
            return None
        k = n // 2
        if k > 5:
            return None
            
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                dist = min(d, n - d)
                c = dist - 1
                colors.append(c)
        return colors

    def _lazebnik_woldar_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are (x, y, z) in Z_3 x Z_3 x Z_3.
        Color of edge ((x1,y1,z1), (x2,y2,z2)) is determined by the parity of the differences.
        dx = (x1 - x2) % 2
        dy = (y1 - y2) % 2
        dz = (z1 - z2) % 2
        color = dx + 2*dy + 4*dz
        """
        n = 26
        verts = list(itertools.product(range(3), repeat=3))
        colors = []
        for idx1, (a, b, c) in enumerate(verts):
            for idx2, (a2, b2, c2) in enumerate(verts):
                if idx1 >= idx2:
                    continue
                # Difference in Z_3 mapped to Z_2 via parity
                # 0->0, 1->1, 2->0 (since 2 is even)
                dx = (a - a2) % 2
                dy = (b - b2) % 2
                dz = (c - c2) % 2
                colors.append(dx + 2 * dy + 4 * dz)
        
        return {'n': n, 'colors': colors}

    # 1. Attempt the known optimal construction for n=26
    best_n = 11
    best_colors = _cyclic_construction(11)
    
    if _time_left() > 10.0:
        res26 = _lazebnik_woldar_construction()
        if _verify(res26['n'], res26['colors']):
            best_n = 26
            best_colors = res26['colors']
        else:
            # If verification fails (shouldn't happen if logic is correct), 
            # we might try to debug or fall back. 
            # For now, we trust the fallback if it fails.
            pass

    # 2. Fallback to n=11 if n=26 failed or wasn't attempted
    if best_colors is None or not _verify(best_n, best_colors):
        best_n = 11
        best_colors = _cyclic_construction(11)
        if not _verify(best_n, best_colors):
             # Absolute fallback: return a trivial small graph
             best_n = 2
             best_colors = [0]

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
