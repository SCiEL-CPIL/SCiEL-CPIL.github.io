import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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
                    common = mask_u & adj[c][v]
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

    def _ag25_construction():
        """
        Affine Plane AG(2,5) construction for n=25.
        Vertices are (x, y) in GF(5)^2.
        Edge color is the slope of the line connecting them.
        Slopes: 0, 1, 2, 3, 4, infinity. That's 6 slopes.
        We have 5 colors. This won't work directly unless we merge slopes.
        Wait, the task says 5 colors.
        The slopes in GF(5) are 0, 1, 2, 3, 4, and infinity.
        If we use 5 colors, we must merge two slopes.
        Let's try merging slope 0 and infinity? Or just use 5 of the 6?
        If we omit one slope, that color class is missing.
        Actually, the standard construction for R_5(C4) >= 25 uses 5 colors.
        How?
        Maybe the vertices are not just (x,y) but something else?
        Or maybe the coloring is different.
        
        Let's check the Lazebnik-Woldar construction for n=26.
        It uses Z_3^3 minus 0.
        
        What about n=25?
        Maybe using GF(5)^2 and mapping slopes to colors is not the way.
        
        Let's stick to the known n=26 construction if it works.
        If n=26 works, we don't need n=25.
        
        Let's implement a proper n=26 construction.
        """
        return None

    def _lw26_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are non-zero vectors in Z_3^3.
        There are 3^3 - 1 = 26 such vectors.
        Color of edge (u, v) is determined by the difference u-v.
        The difference is also a non-zero vector in Z_3^3.
        We need to map the 26 non-zero vectors to 5 colors such that
        no color class contains a C4.
        
        A C4 in the graph of color c corresponds to four distinct vertices
        u, v, w, z such that u-v, v-w, w-z, z-u are all in the set S_c
        of vectors assigned color c.
        Actually, the condition is that the graph G_c with vertices Z_3^3\{0}
        and edges (u,v) if u-v in S_c has no C4.
        
        The standard construction uses a partition of the 26 non-zero vectors
        into 5 sets S_0, ..., S_4 such that each S_i is a difference set
        with specific properties?
        
        Actually, the construction is based on the fact that Z_3^3\{0} can be
        partitioned into 5 sets of size 5? No, 26/5 is not an integer.
        26 = 5*5 + 1.
        
        Let's look at the structure of Z_3^3.
        The non-zero vectors can be grouped by their "norm" or something?
        
        Alternative: The construction uses the fact that the 26 vertices
        can be arranged in 13 pairs of opposites (v, -v).
        
        Let's try a specific known construction.
        One common construction for R_5(C4) >= 26 uses the vertices of Z_3^3 \ {0}.
        The color of edge (u, v) is determined by the pair {u-v, -(u-v)}.
        There are 13 such pairs (since (u-v) and -(u-v) are distinct non-zero vectors).
        So we have 13 "directions".
        We need to assign these 13 directions to 5 colors.
        If we assign 13 directions to 5 colors, some colors will have 3 directions,
        some 2, some 2? 13 = 3+3+3+2+2.
        
        The condition for no C4 is that for any color c, the set of directions D_c
        assigned to c must not contain any "parallelogram" in Z_3^3.
        A parallelogram is u, v, w, z with u-w = v-z (and u-v, v-w, w-z, z-u in D_c?).
        Actually, a C4 is u-v-w-z-u.
        This implies u-w = v-z.
        So if u-w and v-z are both in D_c, and u-v, v-w, w-z, z-u are in D_c,
        then we have a C4.
        
        Wait, the edge colors are determined by the direction of the edge.
        So edge (u,v) has color c if u-v is in D_c.
        A C4 u-v-w-z-u exists if:
        u-v in D_c
        v-w in D_c
        w-z in D_c
        z-u in D_c
        
        Note that z-u = -(u-z).
        So we need u-v, v-w, w-z, u-z all in D_c (up to sign, since D_c is a set of pairs {d, -d}).
        
        Let's try to construct D_c.
        The 13 directions are the 13 pairs {v, -v} for v in Z_3^3 \ {0}.
        
        We can use a greedy or backtracking search to assign the 13 directions to 5 colors
        such that no C4 is formed.
        Since 13 is small, we can try all assignments? 5^13 is too large.
        But we can use symmetry.
        
        Alternatively, there is a known explicit construction.
        Let's try to find one.
        
        One known construction uses the following partition of the 13 directions:
        Let the directions be indexed by the 13 points of the projective plane PG(2,3).
        Wait, Z_3^3 has 26 non-zero vectors, which form 13 lines through the origin.
        These 13 lines form the projective plane PG(2,3).
        
        The condition for no C4 is related to the lines in PG(2,3).
        A C4 in the Cayley graph corresponds to a "parallelogram" in Z_3^3.
        This corresponds to a "quadrangle" in PG(2,3)?
        
        Actually, the Cayley graph on Z_3^3 with connection set S (a set of directions)
        has a C4 if and only if S contains a "difference set" that forms a parallelogram.
        
        Let's try a simple assignment:
        Assign the 13 directions to 5 colors as evenly as possible.
        3, 3, 3, 2, 2.
        
        Let's use a backtracking search to find a valid assignment of the 13 directions to 5 colors.
        """
        # Generate the 13 directions
        vectors = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    vectors.append((x, y, z))
        
        # Group into pairs {v, -v}
        directions = []
        used = set()
        for v in vectors:
            if v in used:
                continue
            neg_v = ((-v[0]) % 3, (-v[1]) % 3, (-v[2]) % 3)
            directions.append((v, neg_v))
            used.add(v)
            used.add(neg_v)
        
        # There should be 13 directions
        assert len(directions) == 13
        
        # We need to assign each direction to a color 0..4.
        # Check if the resulting coloring has no C4.
        
        def check_assignment(assignment):
            # assignment is a list of 13 colors, one for each direction.
            # Build the color for each pair (u, v).
            # Map each vector to its direction index.
            vec_to_dir = {}
            for i, (v, nv) in enumerate(directions):
                vec_to_dir[v] = i
                vec_to_dir[nv] = i
            
            # Build adjacency for each color
            adj = [[0] * 26 for _ in range(5)]
            for i in range(26):
                for j in range(i + 1, 26):
                    u = vectors[i]
                    v = vectors[j]
                    diff = ((u[0] - v[0]) % 3, (u[1] - v[1]) % 3, (u[2] - v[2]) % 3)
                    dir_idx = vec_to_dir[diff]
                    c = assignment[dir_idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            
            # Check for C4
            for c in range(5):
                for u in range(26):
                    mask_u = adj[c][u]
                    for v in range(u + 1, 26):
                        common = mask_u & adj[c][v]
                        if common & (common - 1):
                            return False
            return True
        
        # Backtracking search
        # Symmetry breaking: assign direction 0 to color 0.
        assignment = [0] * 13
        assignment[0] = 0
        
        def backtrack(idx):
            if idx == 13:
                return check_assignment(assignment)
            for c in range(5):
                assignment[idx] = c
                if backtrack(idx + 1):
                    return True
            return False
        
        if backtrack(1):
            # Build the final coloring
            vec_to_dir = {}
            for i, (v, nv) in enumerate(directions):
                vec_to_dir[v] = i
                vec_to_dir[nv] = i
            
            colors = []
            for i in range(26):
                for j in range(i + 1, 26):
                    u = vectors[i]
                    v = vectors[j]
                    diff = ((u[0] - v[0]) % 3, (u[1] - v[1]) % 3, (u[2] - v[2]) % 3)
                    dir_idx = vec_to_dir[diff]
                    colors.append(assignment[dir_idx])
            return colors
        else:
            return None

    # 1. Try known algebraic constructions
    best_n = 11
    best_colors = _cyclic_construction(11)
    if best_colors and _verify(11, best_colors):
        pass # Already valid

    # Try n=26 construction
    if _time_left() > 5.0:
        res26 = _lw26_construction()
        if res26 and _verify(26, res26):
            best_n = 26
            best_colors = res26

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
                # Simple local search
                num_edges = n * (n - 1) // 2
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                
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
                
                max_steps = 2000
                for step in range(max_steps):
                    if _time_left() < 2.0:
                        break
                    
                    if best_badness == 0:
                        break
                    
                    idx = random.randint(0, num_edges - 1)
                    u = 0
                    v = 1
                    # Find u, v for idx
                    # Precompute edge_map for speed
                    # For now, just pick random u, v
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    # Find idx for (u, v)
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            if i == u and j == v:
                                break
                            idx += 1
                    
                    old_c = colors[idx]
                    new_c = random.randint(0, 4)
                    if new_c == old_c:
                        continue
                    
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    
                    new_badness = get_badness()
                    
                    if new_badness < best_badness:
                        best_badness = new_badness
                        colors[idx] = new_c
                    else:
                        adj[new_c][u] &= ~(1 << v)
                        adj[new_c][v] &= ~(1 << u)
                        adj[old_c][u] |= (1 << v)
                        adj[old_c][v] |= (1 << u)
                
                if best_badness == 0:
                    if _verify(n, colors):
                        best_n = n
                        best_colors = colors
                        found = True
                        break
            if found:
                pass
            else:
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
