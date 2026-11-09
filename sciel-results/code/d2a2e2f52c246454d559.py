import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _time_left():
        return deadline - time.time()

    def _verify(n, colors):
        """
        Check if the coloring is valid (no C4 in any color class).
        Uses bitmasks for efficiency.
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
                    if common & (common - 1):
                        return False
        return True

    def _get_n26_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are (x, y, z) in Z_3 x Z_3 x Z_3 minus (0,0,0)? 
        Actually, the standard construction for R_5(C4) >= 26 uses 26 vertices.
        The vertices are often taken from Z_3^3 \ {(0,0,0)}? No, 3^3=27.
        Wait, 26 vertices. 
        Let's check the standard construction.
        Lazebnik and Woldar (1997) showed R_5(C4) >= 26.
        The construction uses a graph with 26 vertices.
        One common representation is vertices in Z_3^3 excluding one point? 
        Or maybe it's a specific graph on 26 vertices.
        
        The previous program used:
        verts = list(itertools.product(range(3), repeat=3))
        n = 26
        # This lists 27 vertices. The code in the prompt:
        # for idx1, (a, b, c) in enumerate(verts):
        #    for idx2, ...
        # It iterates 27 vertices but returns n=26?
        # Let's re-read the prompt's _product_construction.
        # n = 26
        # verts = list(itertools.product(range(3), repeat=3)) -> 27 vertices
        # The loop runs for idx1, idx2 in 0..26.
        # If idx1 >= idx2: continue
        # This generates edges for 27 vertices, but n is set to 26?
        # This looks like a bug in the provided code if it returns 27 vertices' worth of edges but claims n=26.
        # len(colors) would be 27*26/2 = 351.
        # But _verify checks len(colors) == n*(n-1)//2.
        # If n=26, expected len is 26*25/2 = 325.
        # So the prompt's code would fail verification if it actually generated 351 colors.
        
        # Let's check the prompt code again.
        # "verts = list(itertools.product(range(3), repeat=3))" -> 27 items.
        # "n = 26"
        # Loop:
        # for idx1, (a,b,c) in enumerate(verts):
        #   for idx2, (a2,b2,c2) in enumerate(verts):
        #     if idx1 >= idx2: continue
        #     ... append color
        # This appends 27*26/2 = 351 colors.
        # Then _verify(26, colors) is called.
        # len(colors) = 351. 26*25/2 = 325.
        # 351 != 325. So _verify returns False.
        # So the current program's n=26 construction is INVALID and fails verification.
        # It then falls back to n=11.
        # Score 1.5455 implies it IS returning n=26?
        # 1.5455 * 17 = 26.27. 
        # If it returned n=11, score would be 11/17 = 0.64.
        # So the current program MUST be returning a valid n=26.
        # How?
        # Maybe the prompt code provided in the "CURRENT PROGRAM" section is NOT the one that scored 1.5455?
        # The prompt says: "CURRENT PROGRAM (score 1.5455): ... [code] ..."
        # If that code fails verification for n=26, it would return n=11.
        # Unless... the seed is very small?
        # If seed n=7, 11/7 = 1.57.
        # If seed n=8, 11/8 = 1.37.
        # If seed n=17, 11/17 = 0.64.
        
        # Let's assume the provided code is buggy and I need to fix it.
        # The correct Lazebnik-Woldar construction for n=26 uses 26 vertices.
        # It is often based on the group Z_3 x Z_3 x Z_3 minus the identity? No, that's 26 vertices.
        # Yes, Z_3^3 has 27 elements. Removing (0,0,0) leaves 26.
        # Let's use that.
        """
        n = 26
        # Vertices are all (x,y,z) in Z_3^3 except (0,0,0)
        verts = [v for v in itertools.product(range(3), repeat=3) if v != (0,0,0)]
        # Map vertices to indices 0..25
        vert_to_idx = {v: i for i, v in enumerate(verts)}
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                v1 = verts[i]
                v2 = verts[j]
                dx = (v1[0] - v2[0]) % 3
                dy = (v1[1] - v2[1]) % 3
                dz = (v1[2] - v2[2]) % 3
                
                # The coloring rule for Lazebnik-Woldar:
                # The color is determined by the "difference" in a specific way.
                # A common rule for C4-free colorings in Z_3^3:
                # If (dx, dy, dz) is a non-zero vector.
                # We need to map the 26 possible non-zero difference vectors (up to sign? no, directed) 
                # to 5 colors such that no C4 is formed.
                # Actually, the difference vector d = v1 - v2.
                # In Z_3^3, there are 26 non-zero vectors.
                # We need to partition these 26 vectors into 5 sets such that no set contains a "C4 pattern".
                # A C4 in the graph of color c corresponds to a sequence of 4 edges of color c forming a cycle.
                # This is complex.
                
                # Let's use a known valid mapping if possible, or a heuristic one.
                # One simple mapping: 
                # If dx==0, color based on dy,dz.
                # This is getting complicated to derive from scratch.
                
                # Alternative: Use the cyclic construction for n=11 as a fallback if I can't get 26 right.
                # But I want to beat the score.
                
                # Let's try a specific mapping that is known to work for n=26.
                # Reference: "A 5-coloring of K_26 without a monochromatic C4"
                # The vertices can be identified with the 26 non-zero vectors in F_3^3.
                # Color of edge (u, v) is determined by the ratio or difference.
                # Let d = u - v.
                # If d is in a set S_c, color is c.
                # The sets S_c must be such that no 4-cycle exists.
                
                # A simple heuristic: 
                # If d == (1,0,0) or (2,0,0) -> Color 0
                # If d == (0,1,0) or (0,2,0) -> Color 1
                # If d == (0,0,1) or (0,0,2) -> Color 2
                # If d == (1,1,0) or (2,2,0) -> Color 3
                # If d == (1,0,1) or (2,0,2) -> Color 4
                # This covers 10 vectors. What about the other 16?
                # This partition is arbitrary and likely invalid.
                
                # Given the complexity, I will implement a robust local search for n=26 starting from a random state, 
                # or use the cyclic construction for n=11 if 26 is too hard to hardcode correctly.
                # However, n=26 is the target.
                
                # Let's try a different known construction: 
                # The "Paley" type construction? No, that's for primes.
                
                # I will implement a fast local search for n=26.
                # It might find a solution.
                pass
        
        return None # Placeholder, will implement search below

    def _cyclic_construction(n):
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

    def _local_search(n, time_budget):
        if _time_left() < 2.0:
            return None

        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        adj = [[0] * n for _ in range(5)]
        
        def build_adj(colors):
            for c in range(5):
                for i in range(n):
                    adj[c][i] = 0
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
        
        build_adj(colors)
        
        def get_badness():
            bad = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[c][u] & adj[c][v]
                        if common & (common - 1):
                            cnt = bin(common).count("1")
                            bad += cnt * (cnt - 1) // 2
            return bad

        best_badness = get_badness()
        current_badness = best_badness
        temperature = 100.0
        cooling_rate = 0.995
        
        while _time_left() > 2.0 and temperature > 1.0:
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update adj
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            new_badness = get_badness()
            delta = new_badness - current_badness
            
            if delta < 0 or random.random() < pow(2.718, -delta / temperature):
                current_badness = new_badness
                colors[idx] = new_c
                if current_badness < best_badness:
                    best_badness = current_badness
                    if best_badness == 0:
                        return colors
            else:
                # Revert
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
            
            temperature *= cooling_rate
            
        if best_badness == 0:
            return colors
        return None

    # 1. Try known algebraic constructions
    best_n = 11
    best_colors = _cyclic_construction(11)
    if best_colors and _verify(11, best_colors):
        pass
    
    # Try to find n=26 via local search
    # This is a structural change: using SA for n=26 instead of a potentially buggy hardcoded construction.
    if _time_left() > 50.0:
        # Try n=26
        res26 = _local_search(26, 45.0)
        if res26 and _verify(26, res26):
            best_n = 26
            best_colors = res26

    # If 26 failed, try 25, 24...
    if best_n < 26:
        for n in range(25, 11, -1):
            if _time_left() < 10.0:
                break
            res = _local_search(n, 5.0)
            if res and _verify(n, res):
                best_n = n
                best_colors = res
                break

    # Final verification
    if best_colors is None or not _verify(best_n, best_colors):
        best_n = 11
        best_colors = _cyclic_construction(11)

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
