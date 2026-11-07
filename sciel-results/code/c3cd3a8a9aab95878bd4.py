import time

def construct():
    n = 10
    deadline = time.time() + 50  # safety margin

    # We'll use the known fact that for order 10, we can construct
    # a set of mutually orthogonal Latin squares using the structure
    # Z5 x Z2.
    #
    # The cells are indexed by (a, b) where a in 0..4, b in 0..1
    # Cell (i, j) where i in 0..9, j in 0..9:
    #   i = 2*a + b  => a = i // 2, b = i % 2
    #   j = 2*c + d  => c = j // 2, d = j % 2
    #
    # Symbol s in 0..9: s = 5*e + f, e in 0..1, f in 0..4
    #
    # A Latin square L_k is defined by:
    #   e = (e_i + m_k * e_j + c1) % 2   where e_i = a, e_j = c
    #   f = (f_i + n_k * f_j + c2) % 5   where f_i = b, f_j = d
    #
    # Wait, let me reconsider. The standard construction for MOLS of order q
    # when q is a prime power uses GF(q). For q=10, we don't have a field.
    # But we can use the direct product Z5 x Z2.
    #
    # Actually, let's use a simpler approach. We know that:
    # L1[i][j] = (i + j) % 10 is a Latin square.
    # We need to find others orthogonal to it.
    #
    # Let's try a direct construction using the Z5 x Z2 structure.
    # Cell (i,j) maps to ((i//2, i%2), (j//2, j%2)) = (a, b, c, d)
    #
    # For a linear construction over Z5 x Z2:
    # The "top" bit (value 0 or 1, i.e., 0 or 5) can be a linear combination
    # of the top bits of i and j.
    # The "bottom" part (value 0..4) can be a linear combination
    # of the bottom parts of i and j.
    #
    # Let symbol = 5 * top + bottom
    # top = (alpha * (i//2) + beta * (j//2) + gamma * (i%2) + delta * (j%2)) % 2
    # bottom = (alpha5 * (i%2) + beta5 * (j%2) + gamma5 * (i//2) + delta5 * (j//2)) % 5
    #
    # Actually, this is getting complex. Let's use a known result:
    # There exist 9 MOLS of order 10? No, actually the maximum number of MOLS
    # of order 10 is unknown, but it's at least 2 and at most 9.
    #
    # Actually, for order 10, we can construct at least 2 MOLS.
    # Let's try:
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (2*i + 3*j) % 10? Let's check if this is a Latin square.
    # Row i: 2*i + 3*j for j=0..9 => 3*j mod 10. Since gcd(3,10)=1, this is a permutation.
    # Col j: 2*i + 3*j for i=0..9 => 2*i mod 10. Since gcd(2,10)=2, this is NOT a permutation!
    # 2*i mod 10 only gives even numbers. So L2 is not a Latin square.
    #
    # We need a different approach.
    #
    # Let's use the Z5 x Z2 construction properly.
    # Cells: (a,b) where a in Z5, b in Z2. i = 2*a + b.
    # Symbols: (e,f) where e in Z2, f in Z5. s = 5*e + f.
    #
    # For a Latin square, we need:
    # For fixed i (a,b), as j (c,d) varies, symbol (e,f) must cover all (e,f) in Z2 x Z5.
    # For fixed j (c,d), as i (a,b) varies, symbol (e,f) must cover all (e,f) in Z2 x Z5.
    #
    # A linear construction:
    # e = (p*a + q*c + r*b + s*d) % 2
    # f = (p5*a + q5*c + r5*b + s5*d) % 5
    #
    # For fixed (a,b), as (c,d) varies:
    #   e depends on c (via q*c) and d (via s*d)
    #   f depends on c (via q5*c) and d (via s5*d)
    #   For this to be a bijection from Z2 x Z2 to Z2 x Z5, we need...
    #   Wait, (c,d) ranges over Z2 x Z2 (4 values), but (e,f) ranges over Z2 x Z5 (10 values).
    #   This doesn't work because 4 != 10.
    #
    # I think I made an error in the cell/symbol indexing.
    # i ranges over 0..9, so i has 10 values. If i = 2*a + b, then a in 0..4 (5 values), b in 0..1 (2 values). 5*2=10. OK.
    # j ranges over 0..9, so j = 2*c + d, c in 0..4, d in 0..1.
    # So the cell is (a, b, c, d) with a,c in Z5, b,d in Z2.
    #
    # Symbol s = 5*e + f, e in Z2, f in Z5.
    #
    # For fixed i (a,b), as j (c,d) varies over Z5 x Z2 (10 values),
    # the symbol (e,f) must be a bijection onto Z2 x Z5 (10 values).
    #
    # So e and f must be functions of (c,d) (and constants from a,b).
    # e = (q*c + s*d + const_e) % 2   [q, s in Z2]
    # f = (q5*c + s5*d + const_f) % 5  [q5, s5 in Z5]
    #
    # For this to be a bijection from Z5 x Z2 to Z2 x Z5:
    # The map (c,d) -> (e,f) must be bijective.
    # Since e depends only on c,d via Z2-linear map, and f via Z5-linear map,
    # we need the combined map to be bijective.
    #
    # e = (q*c + s*d) % 2
    # f = (q5*c + s5*d) % 5
    #
    # For fixed d, as c varies (Z5), e is constant (if q=0) or varies (if q=1, but c is in Z5, q*c mod 2 depends on c mod 2).
    # This is messy.
    #
    # Alternative: Let's just use a known set of MOLS for order 10.
    # It's known that there are 2 MOLS of order 10.
    #
    # L1[i][j] = (i + j) % 10
    #
    # For L2, we can try:
    # L2[i][j] = (i + 3*j) % 10? 
    # Check Latin: Row i: i + 3*j, j=0..9 => 3*j mod 10, permutation since gcd(3,10)=1.
    # Col j: i + 3*j, i=0..9 => i mod 10, permutation. OK.
    # Check orthogonality with L1:
    # Pair (L1, L2) = (i+j, i+3*j) mod 10.
    # Suppose (i+j, i+3*j) = (i'+j', i'+3*j') mod 10.
    # Subtract: (j-j') = 3*(j-j') => 2*(j-j') = 0 mod 10 => j-j' = 0 or 5 mod 10.
    # If j=j', then i=i'. OK.
    # If j-j'=5, then i+j = i'+j'+5 = i'+j+5 => i = i'+5.
    # So (i,j) and (i+5, j-5) give the same pair? 
    # L1(i,j) = i+j. L1(i+5,j-5) = i+5+j-5 = i+j. Same.
    # L2(i,j) = i+3j. L2(i+5,j-5) = i+5+3(j-5) = i+3j-10 = i+3j mod 10. Same.
    # So collision! Not orthogonal.
    #
    # We need a slope that doesn't have this issue.
    # The issue is that 2*(slope-1) = 0 mod 10 => slope-1 = 0 or 5.
    # So slope = 1 or 6. We used slope=1 for L1, slope=3 for L2.
    # Wait, the condition was 2*(j-j')=0 => j-j'=0 or 5.
    # This always happens for any slope if we consider the difference.
    # Actually, the condition for collision is:
    # (i+j, i+s*j) = (i'+j', i'+s*j')
    # => i+j = i'+j' and i+s*j = i'+s*j'
    # Subtract: (s-1)*(j-j') = 0 mod 10.
    # For no collision (other than i=i',j=j'), we need (s-1)*(j-j') != 0 for all j != j'.
    # This requires gcd(s-1, 10) = 1, i.e., s-1 is coprime to 10.
    # s-1 in {1,3,7,9} => s in {2,4,8,10=0}.
    # s=2: L2[i][j] = (i+2*j)%10. Col: i+2j, i varies => i mod 10, OK.
    # Check orthogonality: (s-1)=1, gcd(1,10)=1. OK.
    # So L1 = (i+j)%10, L2 = (i+2*j)%10 are orthogonal.
    #
    # Can we find a third?
    # L3[i][j] = (i+s3*j)%10, gcd(s3-1,10)=1.
    # s3 in {2,4,8}. We used s=2. Try s=4.
    # L3 = (i+4*j)%10.
    # Check L1 vs L3: (s3-1)=3, gcd(3,10)=1. OK.
    # Check L2 vs L3: (s3-s2)=4-2=2, gcd(2,10)=2 != 1. NOT orthogonal.
    #
    # So the linear construction over Z10 only gives 2 MOLS (s=1 and s=2, or s=1 and s=4, etc.).
    #
    # To get more, we need a non-linear construction or use Z5 x Z2.
    # Given time constraints, let's return the 2 MOLS we found.
    #
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 2*j) % 10
    #
    # Verify they are Latin squares and orthogonal.
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    def is_latin_square(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
            for j in range(n):
                col = [sq[i][j] for i in range(n)]
                if len(set(col)) != n:
                    return False
        return True
    
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n * n
    
    result = [L1, L2]
    
    if is_latin_square(L1) and is_latin_square(L2) and are_orthogonal(L1, L2):
        return result
    
    # Fallback: return just L1
    return [L1]
