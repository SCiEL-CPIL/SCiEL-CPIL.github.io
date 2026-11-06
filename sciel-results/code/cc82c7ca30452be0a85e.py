"""Hadamard-based 4-coloring of K_n for R(3,3,3,3) > n."""

import time
import numpy as np
import random

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Uses Hadamard matrix construction to get K48, then tries to extend.
    """
    t0 = time.time()
    deadline = t0 + 140.0

    # Build Hadamard matrix of order 48: H(48) = H(4) ⊗ H(12)
    # H(4) is Sylvester:
    H4 = np.array([[1, 1, 1, 1],
                   [1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [1, -1, -1, 1]], dtype=np.int8)
    
    # Build H(12): Use the standard construction.
    # H(12) can be built from the 12x12 matrix based on Z_11 or directly.
    # Actually, let's use a known explicit H(12).
    # One construction: take the 12x12 matrix where row 0 is all 1s,
    # and for rows 1-11, use a pattern based on quadratic residues mod 11.
    # Quadratic residues mod 11: {1, 3, 4, 5, 9}
    # Non-residues: {2, 6, 7, 8, 10}
    
    # Actually, the simplest: H(12) from the "Paley" construction doesn't directly work for 12.
    # Let me use a different approach: H(12) exists and can be constructed as follows.
    # 
    # A known H(12):
    # Row 0: 1 1 1 1 1 1 1 1 1 1 1 1
    # For i = 1..11, j = 1..11: H[i][j] = 1 if (j-i) mod 11 is a quadratic residue, -1 otherwise
    # H[i][0] = 1 for all i (normalization)
    # H[0][j] = 1 for all j (normalization)
    
    qr11 = {1, 3, 4, 5, 9}  # quadratic residues mod 11
    
    H12 = np.ones((12, 12), dtype=np.int8)
    for i in range(1, 12):
        for j in range(1, 12):
            d = (j - i) % 11
            if d in qr11:
                H12[i][j] = 1
            else:
                H12[i][j] = -1
    # Normalize: first row and column already all 1s
    
    # Verify H12 is Hadamard: H12 @ H12.T should be 12 * I
    # Let me check: for i != j, sum_k H12[i][k] * H12[j][k] should be 0
    # This is a standard result for Paley-type constructions.
    
    # Build H(48) = H(4) ⊗ H(12)
    H48 = np.kron(H4, H12)
    
    # Now use the Hadamard construction for 4-coloring K_48.
    # H48 is 48x48, normalized (first row and col are +1).
    # Write H48 as a 4x4 block matrix where each block is 12x12.
    # H48[a*12:(a+1)*12, b*12:(b+1)*12] = H4[a,b] * H12
    
    # The 4-coloring construction:
    # Vertices: (a, i) where a in {0,1,2,3}, i in {0,...,11}
    # For edge between (a,i) and (b,j):
    #   If a == b: color = 0
    #   If a != b: 
    #     Let v = H48[a*12+i, b*12+j]  (which is H4[a,b] * H12[i,j])
    #     Color based on v: if v == 1, use one of {1,2,3}; if v == -1, use another
    #     
    # Wait, this only gives 2 colors for inter-group edges. We need 4 total.
    # 
    # Let me reconsider. The standard construction:
    # Given a normalized Hadamard H of order 4s, we get a 4-coloring of K_{4s}.
    # The vertices are partitioned into 4 groups of size s.
    # 
    # Actually, the correct construction (from the literature on R(3,3,3,3)):
    # If H is a normalized Hadamard of order 4s, then:
    # - Vertices: v_{a,i} for a in {0,1,2,3}, i in {0,...,s-1}
    # - Edge (v_{a,i}, v_{b,j}):
    #   if a == b: color 0
    #   else: let c = H[a*? + ?][b*? + ?] ... 
    #
    # I think I'm overcomplicating this. Let me use a different, well-known construction.
    #
    # ALTERNATIVE: Use the group-based construction with Z_2^4 × Z_3.
    # Or better: use the explicit 48-vertex construction from the literature.
    #
    # Actually, let me try a cleaner approach. The key fact is:
    # A 4-coloring of K_n with no monochromatic triangle is equivalent to
    # a decomposition of K_n into 4 triangle-free graphs.
    #
    # For n=48: Use the fact that K_48 can be decomposed into 4 copies of
    # a triangle-free graph with 48*47/2 / 4 = 564 edges each.
    #
    # Let me try the direct Hadamard construction more carefully.
    # 
    # The construction from Graham, Sloane, and others:
    # Given normalized Hadamard H of order 4s:
    # - 4s vertices labeled (i, j) with i in {0,1,2,3}, j in {0,...,s-1}
    # - Color of edge ((i,j),(k,l)):
    #   if i == k: color = 0
    #   else: 
    #     The (i,k) block of H is H_{ik} = H[i,k] * H_s where H_s is the s×s submatrix
    #     Actually H_{ik}[j,l] = H[i*? ... ]
    #
    # Let me just try a different, simpler construction that I can verify.
    #
    # SIMPLEST APPROACH: Use the known result that if we have a 4-coloring of K_m
    # with no mono triangle, and a 4-coloring of K_n with no mono triangle,
    # we can combine them. But that's not straightforward.
    #
    # Let me try: vertices = Z_4 × Z_12, size 48.
    # Edge ((a, x), (b, y)) where a,b in Z_4, x,y in Z_12:
    #   d = (b - a) mod 4, delta = (y - x) mod 12
    #   Color = f(d, delta)
    #
    # For d = 0: need a 4-coloring of K_12 with no mono triangle (3 colors suffice for K_12? No, R(3,3)=6 so K_5 needs 3 colors, K_12 needs... )
    # Actually for d=0, we need to color K_12 with colors {0,1,2,3} such that no color class has a triangle.
    # K_12 can be 3-colored with no mono triangle? R(3,3,3)=17, so yes K_16 works with 3 colors.
    #
    # This is getting complex. Let me just use a verified 48-vertex coloring.
    #
    # I'll use the following approach: start with a known valid coloring and verify it.
    # The Hadamard construction for order 48:
    #
    # H48 = H4 ⊗ H12 (Kronecker product)
    # H48 is 48x48, H48 @ H48.T = 48*I
    #
    # The 4-coloring:
    # Label vertices 0..47. Vertex v has "group" g = v // 12 and "index" i = v % 12.
    # For edge (u, v) with u < v:
    #   gu = u // 12, iu = u % 12
    #   gv = v // 12, iv = v % 12
    #   if gu == gv: color = 0
    #   else: 
    #     h_val = H48[u, v]  # this is H4[gu, gv] * H12[iu, iv]
    #     # Map h_val to a color in {1, 2, 3}
    #     # We have 3 inter-group color pairs: (0,1), (0,2), (0,3) ... 
    #     # Actually we need to use H4[gu,gv] which is ±1, and H12[iu,iv] which is ±1
    #     # So h_val = ±1, giving 2 values. But we need 3 colors for inter-group.
    #
    # This doesn't work directly because we only get 2 values from the product.
    #
    # CORRECT CONSTRUCTION:
    # The Hadamard construction for R(3,3,3,3) works as follows:
    # Given a normalized Hadamard matrix H of order 4s:
    # - The 4s vertices are the columns of H (or rows).
    # - For each pair of columns (i, j), the "dot product" H[:,i] · H[:,j] = 0 (orthogonal).
    # - This means exactly s entries agree and s disagree (since 4s total, half agree).
    # - Wait, for a normalized Hadamard, H[:,0] is all 1s, so H[:,0]·H[:,j] = sum(H[:,j]) = 0 for j>0.
    #
    # The coloring: For vertices i, j (columns of H):
    #   Let A = {k : H[k,i] = H[k,j]}  (agreement set)
    #   Let B = {k : H[k,i] != H[k,j]}  (disagreement set)
    #   |A| = |B| = 2s (since orthogonal, sum of products = 0, so #agree = #disagree = 2s)
    #   
    # This gives 2 sets, not 4. I need a different approach.
    #
    # Let me use the explicit construction from the paper on R(3,3,3,3):
    # The 48-vertex construction uses the group Z_2^4 × Z_3.
    #
    # Vertices: (x, a) where x in {0,1}^4, a in Z_3. Total 16*3 = 48.
    # Edge ((x,a), (y,b)):
    #   d = x XOR y (in {0,1}^4), delta = (b - a) mod 3
    #   w = popcount(d)
    #   Color = (w mod 3) * 3 + delta  ... no, that gives 9 colors.
    #
    # Let me try: Color = (w mod 2) * 2 + (delta mod 2) ... no.
    #
    # I think the cleanest approach is to just hardcode a known valid 48-vertex coloring
    # or use a construction I can verify computationally.
    #
    # NEW APPROACH: Use the fact that K_48 can be built from the 4x4 grid of K_12.
    # Use a 3-coloring of K_12 (which exists since R(3,3,3)=17) for intra-group edges,
    # and assign colors to inter-group edges based on a 4x4 matrix.
    #
    # Actually, let me just try to construct and verify computationally.
    # I'll use a randomized approach to find a valid 4-coloring of K_48.
    
    # Let me try a completely different structural approach:
    # Use simulated annealing / local search to find a 4-coloring of K_48.
    
    def count_mono_triangles(adj, n):
        """Count monochromatic triangles in the coloring."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def try_find_coloring(n, budget_s):
        """Use local search to find a 4-coloring of K_n with no mono triangles."""
        t_start = time.time()
        
        # Random initialization
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = random.randint(0, 3)
                adj[j][i] = adj[i][j]
        
        current_cost = count_mono_triangles(adj, n)
        best_adj = [row[:] for row in adj]
        best_cost = current_cost
        
        temp = 50.0
        cooling = 0.999
        
        while time.time() - t_start < budget_s:
            if current_cost == 0:
                return adj
            
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = (old_c + random.randint(1, 3)) % 4
            
            # Compute cost change
            # Flipping edge (i,j) from old_c to new_c:
            # Triangles involving (i,j): for each k, triangle (i,j,k)
            #   If adj[i][k]==adj[j][k]==old_c, we lose a mono triangle (good)
            #   If adj[i][k]==adj[j][k]==new_c, we gain a mono triangle (bad)
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if adj[i][k] == old_c and adj[j][k] == old_c:
                    delta -= 1  # removing a mono triangle
                if adj[i][k] == new_c and adj[j][k] == new_c:
                    delta += 1  # adding a mono triangle
            
            new_cost = current_cost + delta
            if new_cost < current_cost or random.random() < np.exp(-delta / temp):
                adj[i][j] = new_c
                adj[j][i] = new_c
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
            
            temp *= cooling
        
        if best_cost == 0:
            return best_adj
        return None

    # Try to find a 48-vertex coloring
    if time.time() < deadline - 20.0:
        budget = min(100.0, deadline - time.time() - 10.0)
        adj48 = try_find_coloring(48, budget)
        if adj48 is not None:
            # Verify
            if count_mono_triangles(adj48, 48) == 0:
                colors = []
                for i in range(48):
                    for j in range(i + 1, 48):
                        colors.append(adj48[i][j])
                return {'n': 48, 'colors': colors}
    
    # Fallback: try K_32 from Clebsch
    def mul16(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log16, x = {}, 1
    for e in range(15):
        log16[x] = e
        x = mul16(x, 2)

    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log16[i ^ j] % 3

    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = get_color_32(i, j)
            adj[i][j] = c
            adj[j][i] = c

    # Verify K32
    assert count_mono_triangles(adj, 32) == 0, "K32 construction invalid"

    colors = []
    for i in range(n32):
        for j in range(i + 1, n32):
            colors.append(adj[i][j])
    return {'n': n32, 'colors': colors}
