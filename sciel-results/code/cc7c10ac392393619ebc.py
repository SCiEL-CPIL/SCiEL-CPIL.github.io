import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Known record: n=29 via Paley graph on 29 vertices (29 = 4*7+1, 7 is a prime, -1 is a square mod 7)
    # Actually, 29 = 4*7+1. Let's verify: p=29, p ≡ 1 mod 4. Yes.
    # Paley graph construction:
    # Vertices 0..28.
    # Edge (i,j) is color 0 if (j-i) mod 29 is a quadratic residue mod 29, else color 1.
    # Quadratic residues mod 29: {x^2 mod 29 for x in 1..28}
    
    def paley_coloring(n):
        if n % 4 != 1:
            return None
        # Check if n is prime? Paley graph works for any n ≡ 1 mod 4 that is a prime or prime power.
        # For n=29, it is prime.
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}

    # Verify if a coloring is valid (no K2,2,2 in either color)
    def is_valid(n, colors):
        # Build adjacency bitsets for both colors
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        # Check for K2,2,2 in each color
        for color in [0, 1]:
            adj = adj0 if color == 0 else adj1
            # K2,2,2 is the octahedron graph.
            # It consists of 6 vertices partitioned into 3 pairs (A, B, C).
            # All edges between A and B, A and C, B and C exist in the color class.
            # No edges within A, within B, or within C in the color class.
            
            # Efficient check:
            # Iterate over all pairs (u, v) that are NOT connected in this color (potential pair in K222).
            # If (u, v) is not an edge in color, they can be in the same pair.
            # Let N(u) be neighbors of u in color.
            # Let N(v) be neighbors of v in color.
            # Common neighbors C = N(u) & N(v).
            # Any vertex in C is connected to both u and v.
            # We need to find 4 vertices in C: x, y, z, w
            # such that {x, y} is an independent set in color (not connected)
            # {z, w} is an independent set in color (not connected)
            # and all edges between {x, y} and {z, w} exist in color.
            
            # Optimization:
            # For each non-edge (u, v) in color:
            #   C = N(u) & N(v)
            #   For each pair (x, y) in C that is a non-edge in color:
            #       Let R = C & ~N(x) & ~N(y)  # Vertices in C not connected to x or y?
            #       No, we need vertices z, w in C such that:
            #       1. z, w are not connected to each other (non-edge in color)
            #       2. z is connected to x and y
            #       3. w is connected to x and y
            #       
            #       So z and w must be in C ∩ N(x) ∩ N(y).
            #       Let S = C & N(x) & N(y).
            #       We need to find a non-edge (z, w) in S.
            
            # This is O(n^2 * n^2 * n^2) worst case, but with bitsets it's fast.
            # n=29, n^2=841. Pairs ~ 4000.
            # For each pair, C size ~ 10-20. Pairs in C ~ 100.
            # S size ~ 5-10. Pairs in S ~ 50.
            # Total ~ 4000 * 100 * 50 = 20M ops. Might be slow in Python if not careful.
            
            # Alternative: Precompute non-edges for each color?
            # non_adj0[i] = bitmask of vertices NOT connected to i in color 0 (excluding i itself).
            # non_adj1[i] = bitmask of vertices NOT connected to i in color 1.
            
            non_adj = [0] * n
            for i in range(n):
                # All vertices except i
                all_mask = (1 << n) - 1
                all_mask &= ~(1 << i)
                non_adj[i] = all_mask & ~adj[i]
            
            # Iterate over potential pairs (u, v) in the same part of K222.
            # They must be a non-edge in this color.
            for u in range(n):
                # v > u to avoid duplicates
                candidates_v = non_adj[u] & ((1 << n) - 1)
                candidates_v &= ~((1 << (u + 1)) - 1)  # v > u
                
                # Iterate over set bits in candidates_v
                cv = candidates_v
                while cv:
                    lsb = cv & (-cv)
                    v = lsb.bit_length() - 1
                    cv ^= lsb
                    
                    # C = N(u) & N(v)
                    C = adj[u] & adj[v]
                    # Remove u and v from C (they are not in N(u) or N(v) anyway if simple graph, but just in case)
                    C &= ~(1 << u)
                    C &= ~(1 << v)
                    
                    # We need to find 4 vertices in C forming a K2,2.
                    # i.e., two non-edges (x,y) and (z,w) in C, with all cross-edges present.
                    # This is equivalent to finding a K2,2 in the subgraph induced by C,
                    # where the "edges" of the K2,2 are the non-edges of the color graph?
                    # No.
                    # The vertices x,y,z,w are in C.
                    # (x,y) is a non-edge in color.
                    # (z,w) is a non-edge in color.
                    # x-z, x-w, y-z, y-w are edges in color.
                    
                    # Let's iterate over non-edges (x,y) in C.
                    # x, y in C, and (x,y) is a non-edge in color.
                    # So y must be in non_adj[x] & C.
                    
                    cx = C
                    while cx:
                        lsb_x = cx & (-cx)
                        x = lsb_x.bit_length() - 1
                        cx ^= lsb_x
                        
                        # y in C, y > x, and (x,y) is non-edge in color
                        cy = non_adj[x] & C
                        cy &= ~((1 << (x + 1)) - 1)  # y > x
                        cy &= ~(1 << x)
                        
                        cy_iter = cy
                        while cy_iter:
                            lsb_y = cy_iter & (-cy_iter)
                            y = lsb_y.bit_length() - 1
                            cy_iter ^= lsb_y
                            
                            # Now find z, w in C \ {x, y} such that:
                            # (z,w) is non-edge in color
                            # z, w connected to x and y in color
                            
                            # S = C & N(x) & N(y)
                            S = C & adj[x] & adj[y]
                            S &= ~(1 << x)
                            S &= ~(1 << y)
                            
                            # Check if S contains a non-edge (z, w)
                            # z, w in S, (z,w) is non-edge in color.
                            sz = S
                            while sz:
                                lsb_z = sz & (-sz)
                                z = lsb_z.bit_length() - 1
                                sz ^= lsb_z
                                
                                # w in S, w > z, (z,w) non-edge
                                sw = non_adj[z] & S
                                sw &= ~((1 << (z + 1)) - 1)
                                sw &= ~(1 << z)
                                
                                if sw:
                                    # Found K2,2,2
                                    return False
        
        return True

    # Try n=29 Paley
    result_29 = paley_coloring(29)
    if result_29:
        # Verify to be sure
        if is_valid(29, result_29['colors']):
            return result_29
        else:
            print("Paley 29 invalid?", file=sys.stderr)

    # Fallback to n=17 Paley
    result_17 = paley_coloring(17)
    if result_17:
        if is_valid(17, result_17['colors']):
            return result_17

    # Fallback to n=1
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
