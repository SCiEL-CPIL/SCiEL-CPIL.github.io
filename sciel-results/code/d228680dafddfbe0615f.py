import time
import itertools
import random

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
                    # (u, x, v, y, u) where x,y are common neighbors.
                    # Wait, a C4 is u-x-v-y-u.
                    # The condition "no two vertices with two common neighbours" means:
                    # For any pair (u, v), the number of common neighbors in color c must be <= 1.
                    # If count(common) > 1, there is a C4.
                    # common & (common - 1) checks if there are at least 2 bits set.
                    if common & (common - 1):
                        return False
        return True

    def _cyclic_construction(n):
        """
        Cyclic construction for odd n.
        Colors edge (i, j) based on circular distance.
        Valid if n is odd and we use n//2 colors? No, we have 5 colors.
        For n=11, distances 1..5. 
        For n=13, distances 1..6. We have 5 colors. 
        The standard cyclic construction for R_5(C4) usually requires n=2k+1 and k colors?
        Actually, the seed program uses n=11 (k=5) with 5 colors.
        Distance d in 1..5. Color = d-1.
        If n=13, distances 1..6. We can map distance 6 to some color, but it might create C4s?
        In Z_n, a C4 in distance d implies 2d = 0 or 2d = n or something similar?
        Actually, for cyclic distance coloring, a C4 exists if there are i, j, k, l such that
        dist(i,j)=dist(j,k)=dist(k,l)=dist(l,i)=d.
        This implies i, i+d, i+2d, i+3d form a cycle.
        If 2d is not 0 mod n, it's a 4-cycle?
        For n=11, d=1..5. 2d is 2,4,6,8,10. None are 0.
        So no 4-cycles of same distance?
        Wait, a C4 in the graph of distance d edges is a cycle of length 4.
        In Z_11, the graph of distance d is a Hamiltonian cycle (since gcd(d,11)=1).
        A Hamiltonian cycle of length 11 contains no C4.
        So for any prime n=2k+1, the cyclic coloring with k colors is valid?
        We have 5 colors. So max k=5. n=11.
        If we try n=13, k=6. We need to map 6 distances to 5 colors.
        If we merge two distances, say 5 and 6, does it create a C4?
        The union of two Hamiltonian cycles might contain a C4.
        So cyclic construction likely only works for n=11 with 5 colors directly.
        """
        if n % 2 == 0:
            return None
        k = n // 2
        if k > 5:
            # Cannot map >5 distances to 5 colors without merging, which is risky
            # We can try merging the largest distance with the smallest?
            # Or just return None to rely on other methods.
            pass
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                dist = min(d, n - d)
                # For n=11, dist is 1..5. Map to 0..4.
                # For n=13, dist is 1..6. 
                # If we simply take dist % 5, dist=5->0, dist=6->1.
                # This might work? Let's test validity in _verify.
                # But we generate candidates and verify.
                c = dist - 1
                if c >= 5:
                    c = c % 5 # Fallback mapping
                colors.append(c)
        return colors

    def _product_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are (x, y, z) in Z_3 x Z_3 x Z_3.
        Color of edge ((x1,y1,z1), (x2,y2,z2)) is determined by:
        dx = (x1 - x2) % 2
        dy = (y1 - y2) % 2
        dz = (z1 - z2) % 2
        color = dx + 2*dy + 4*dz
        
        Wait, the inspiration code used:
        dx = (a - a2) % 2
        ...
        This maps differences in Z_3 to Z_2?
        0->0, 1->1, 2->0.
        So it's a homomorphism Z_3 -> Z_2? No, 1+1=2->0. 1+2=3->0.
        It's not a group homomorphism.
        Let's check the logic in inspiration 2.
        
        Actually, the standard construction for R_5(C4) >= 26 uses Z_3^3.
        The coloring is often defined using the field GF(3) or similar.
        Let's trust the inspiration code which claims to verify it.
        """
        n = 26
        verts = list(itertools.product(range(3), repeat=3))
        colors = []
        for idx1, (a, b, c) in enumerate(verts):
            for idx2, (a2, b2, c2) in enumerate(verts):
                if idx1 >= idx2:
                    continue
                # Difference in Z_3
                dx = (a - a2) % 3
                dy = (b - b2) % 3
                dz = (c - c2) % 3
                
                # Map Z_3 to {0,1} for color bits?
                # 0 -> 0
                # 1 -> 1
                # 2 -> 0 (since -1)
                # This is essentially abs(diff) % 2?
                # 1->1, 2->1? No.
                # Inspiration 2 code:
                # dx = (a - a2) % 2
                # This is just taking the parity of the difference?
                # a, a2 in {0,1,2}.
                # a=0, a2=1 -> diff=-1 -> %2 = 1.
                # a=1, a2=0 -> diff=1 -> %2 = 1.
                # a=0, a2=2 -> diff=-2 -> %2 = 0.
                # a=2, a2=0 -> diff=2 -> %2 = 0.
                # a=1, a2=2 -> diff=-1 -> %2 = 1.
                # a=2, a2=1 -> diff=1 -> %2 = 1.
                # So it maps distance 1 in Z_3 to 1, distance 2 in Z_3 to 0?
                # Wait, distance 2 is equivalent to -1.
                # So it distinguishes +1 and -1?
                # (a - a2) % 2:
                # 1-0=1->1
                # 0-1=-1->1
                # So it does NOT distinguish +1 and -1.
                # It maps {1, 2} (non-zero) to 1?
                # 2 % 2 = 0.
                # So 2 maps to 0.
                # 1 maps to 1.
                # 0 maps to 0.
                # So it maps: 0->0, 1->1, 2->0.
                
                dx = (a - a2) % 2
                dy = (b - b2) % 2
                dz = (c - c2) % 2
                colors.append(dx + 2 * dy + 4 * dz)
        
        return {'n': n, 'colors': colors}

    def _local_search(n, time_budget):
        """
        Simple hill climbing / simulated annealing style search.
        """
        if _time_left() < 2.0:
            return None

        num_edges = n * (n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge indices to (u, v)
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Build adj masks for initial state
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
                        # Count bits > 1
                        if common & (common - 1):
                            # Approximate badness: number of pairs of common neighbors
                            # Just count bits for now
                            cnt = bin(common).count("1")
                            bad += cnt * (cnt - 1) // 2
            return bad

        best_badness = get_badness()
        
        max_steps = 5000
        for step in range(max_steps):
            if _time_left() < 2.0:
                break
            
            if best_badness == 0:
                return colors
            
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            
            old_c = colors[idx]
            
            # Try changing to a different color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update adj
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            # Calculate new badness (expensive, but n is small)
            # Optimization: Only check pairs involving u or v?
            # For now, full check is O(n^2 * 5).
            # For n=12, ~700 ops. 5000 steps -> 3.5M ops. OK.
            
            new_badness = get_badness()
            
            if new_badness < best_badness:
                best_badness = new_badness
                colors[idx] = new_c
            else:
                # Revert
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                # Keep old color (or random restart if stuck too long?)
                # Simple hill climbing gets stuck.
                # Let's add a random restart if no improvement in 1000 steps?
                # Or just accept worse moves with low prob?
                # Given time constraints, simple hill climbing is safer.
                
        if best_badness == 0:
            return colors
        return None

    # 1. Try known algebraic constructions
    best_n = 11
    best_colors = _cyclic_construction(11)
    if best_colors and _verify(11, best_colors):
        pass # Already valid
    
    # Try n=26 construction
    if _time_left() > 5.0:
        res26 = _product_construction()
        if _verify(res26['n'], res26['colors']):
            best_n = 26
            best_colors = res26['colors']
    
    # Try to find n=12..25 via local search?
    # If we have 26, we don't need 12-25.
    # If we don't have 26, try 12, 13...
    
    if best_n < 26:
        for n in range(12, 26):
            if _time_left() < 10.0:
                break
            
            # Try a few restarts for each n
            found = False
            for attempt in range(3):
                if _time_left() < 5.0:
                    break
                res = _local_search(n, 5.0)
                if res:
                    if _verify(n, res):
                        best_n = n
                        best_colors = res
                        found = True
                        break
            if found:
                # If we found n, try n+1?
                # Or break to save time for verification?
                # Let's continue to see if we can go higher.
                pass
            else:
                # If we failed for n, maybe n+1 is harder?
                # But sometimes structure allows jumps.
                # However, local search is weak.
                # Let's stop if we fail a couple of times to avoid wasting time.
                # Actually, if we failed 12, 13 is likely to fail.
                break

    # Final verification
    if best_colors is None or not _verify(best_n, best_colors):
        # Fallback to 11
        best_n = 11
        best_colors = _cyclic_construction(11)

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
