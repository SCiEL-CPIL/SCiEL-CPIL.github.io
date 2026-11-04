import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    terms = []  # list of (u, v, w) triples
    
    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are the top-left offsets in the full matrices.
        The 2x2 blocks span:
          A: rows [i_off, i_off+1], cols [j_off, j_off+1]
          B: rows [j_off, j_off+1], cols [k_off, k_off+1]
          C: rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        result = []
        
        # M1 = (A11+A22)*(B11+B22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M2 = (A21+A22)*B11
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u, v, w))
        
        # M3 = A11*(B12-B22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M4 = A22*(B21-B11)
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M5 = (A11+A12)*B22
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u, v, w))
        
        # M6 = (A21-A11)*(B11+B12)
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M7 = (A12-A22)*(B21+B22)
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u, v, w))
        
        return result
    
    def strassen_4x4_4x4():
        """
        Recursive Strassen for 4x4 * 4x4 multiplication.
        Uses 7 products of 2x2*2x2, each decomposed with 7-term Strassen.
        Total: 49 terms.
        
        The 7 Strassen products for 4x4*4x4:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        Where blocks are 2x2:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        
        For each product, we need to decompose a 2x2*2x2 multiplication
        where the input matrices may have entries from multiple original blocks.
        """
        result = []
        
        # Define block positions
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        
        # M1 = (A11+A22)*(B11+B22)
        # A' has entries: A[i,j] for i,j in {0,1} from A11, plus A[i+2,j+2] from A22
        # B' has entries: B[j,k] for j,k in {0,1} from B11, plus B[j+2,k+2] from B22
        # This is a 2x2*2x2 where:
        #   A'[0,0] = A[0,0] + A[2,2]
        #   A'[0,1] = A[0,1] + A[2,3]
        #   A'[1,0] = A[1,0] + A[3,2]
        #   A'[1,1] = A[1,1] + A[3,3]
        #   B'[0,0] = B[0,0] + B[2,2]
        #   B'[0,1] = B[0,2] + B[2,3]
        #   B'[1,0] = B[1,0] + B[3,2]
        #   B'[1,1] = B[1,2] + B[3,3]
        #
        # In tensor decomposition, for each Strassen term, we set u and v
        # to have non-zero entries at the appropriate positions.
        
        # For M1, the "virtual" 2x2 A matrix maps:
        #   virtual (0,0) -> physical (0,0) and (2,2)
        #   virtual (0,1) -> physical (0,1) and (2,3)
        #   virtual (1,0) -> physical (1,0) and (3,2)
        #   virtual (1,1) -> physical (1,1) and (3,3)
        #
        # Similarly for B.
        #
        # We can generate the 7 Strassen terms for the virtual 2x2*2x2,
        # then expand each term's u and v vectors to the physical positions.
        
        def expand_strassen_term(u_virt, v_virt, w_virt, a_map, b_map):
            """
            Expand a virtual 2x2 Strassen term to physical positions.
            a_map: dict from virtual (i,j) to list of physical (i,j) positions
            b_map: dict from virtual (j,k) to list of physical (j,k) positions
            w_virt: already in physical C positions (2x2 block)
            """
            u_phys = [0] * (n * m)
            v_phys = [0] * (m * p)
            w_phys = [0] * (p * n)
            
            # Expand u
            for (vi, vj), val in u_virt.items():
                for (pi, pj) in a_map.get((vi, vj), []):
                    u_phys[pi * m + pj] += val
            
            # Expand v
            for (vj, vk), val in v_virt.items():
                for (pj, pk) in b_map.get((vj, vk), []):
                    v_phys[pj * p + pk] += val
            
            # w is already in physical positions for the 2x2 C block
            # But we need to map virtual C positions to physical C positions
            # For M1, C' = C11 + C22, so:
            #   virtual C(0,0) -> physical (0,0) and (2,2)
            #   virtual C(0,1) -> physical (0,1) and (2,3)
            #   virtual C(1,0) -> physical (1,0) and (3,2)
            #   virtual C(1,1) -> physical (1,1) and (3,3)
            # Actually, w_virt is already specified in terms of the 2x2 block output,
            # so we need a c_map too.
            
            return u_phys, v_phys, w_phys
        
        # This approach is getting complex. Let me use a simpler method:
        # Directly generate the 49 terms by combining the 7 outer Strassen
        # products with the 7 inner Strassen terms.
        
        # For each of the 7 outer products, define:
        #   a_positions: list of (i, j) pairs with signs for A side
        #   b_positions: list of (j, k) pairs with signs for B side
        #   c_positions: list of (k, i) pairs with signs for C side
        #
        # Then for each of the 7 inner Strassen terms, combine them.
        
        # Actually, the cleanest way: for each outer product, the "A" and "B"
        # are 2x2 matrices whose entries are linear combinations of original
        # entries. We can represent this as a 2x2 matrix of coefficients.
        
        # Let me just directly write out all 49 terms.
        
        # Define the 7 outer products with their A and B coefficient matrices
        # Each is a 2x2 matrix where entry (r,c) gives the coefficient for
        # A[2r + ir, 2c + jc] where ir,jc are the block offsets.
        
        # This is getting too complex. Let me use the 56-term approach which
        # is simpler and still gives a good score, or find a cleaner way.
        
        # Actually, let me just use the 8-block approach (56 terms) which is
        # straightforward and correct, plus schoolbook for the last 3 columns.
        # Total: 56 + 48 = 104 terms.
        
        # But wait, the inspiration says 49 + 48 = 97. Let me try to implement
        # the 49-term version correctly.
        
        # Key insight: For M1 = (A11+A22)*(B11+B22), the product is:
        # sum_{i,j in {0,1}} (A[i,j] + A[i+2,j+2]) * (B[j,k] + B[j+2,k+2])
        # = sum_{i,j} A[i,j]*B[j,k] + sum_{i,j} A[i,j]*B[j+2,k+2]
        #   + sum_{i,j} A[i+2,j+2]*B[j,k] + sum_{i,j} A[i+2,j+2]*B[j+2,k+2]
        #
        # This is the sum of 4 separate 2x2*2x2 multiplications!
        # But Strassen says we can do it in 7 terms, not 4*7=28.
        #
        # The trick is that the 7 Strassen terms for the "combined" 2x2*2x2
        # automatically account for the cross terms.
        
        # I think the cleanest implementation is to note that:
        # M1 = (A11+A22)*(B11+B22) can be decomposed by applying the 7-term
        # Strassen to a 2x2*2x2 where:
        #   A'[r,c] = A[2r,2c] + A[2r+2,2c+2]
        #   B'[r,c] = B[2r,2c] + B[2r+2,2c+2]
        #
        # In the tensor decomposition, each of the 7 inner Strassen terms
        # has u and v vectors that are sums of contributions from the
        # "diagonal" blocks.
        
        # Let me just implement it by generating the 7 terms for each outer
        # product, where for each term I specify which (i,j) and (j,k) positions
        # are active.
        
        # I'll define a helper that takes:
        #   a_terms: list of (i, j, sign) for A side
        #   b_terms: list of (j, k, sign) for B side  
        #   c_terms: list of (k, i, sign) for C side
        # And generates one rank-1 term.
        
        # For the 7 inner Strassen terms of a 2x2*2x2:
        # Term 1: A: (0,0),(1,1); B: (0,0),(1,1); C: (0,0),(1,1)
        # Term 2: A: (1,0),(1,1); B: (0,0); C: (0,1),(1,1) with signs +1,-1
        # Term 3: A: (0,0); B: (0,1),(1,1); C: (1,0),(1,1)
        # Term 4: A: (1,1); B: (1,0),(0,0); C: (0,0),(0,1)
        # Term 5: A: (0,0),(0,1); B: (1,1); C: (1,0),(0,0) with signs +1,-1
        # Term 6: A: (1,0),(0,0); B: (0,0),(0,1); C: (1,1)
        # Term 7: A: (0,1),(1,1); B: (1,0),(1,1); C: (0,0)
        
        # For M1, the virtual positions map to:
        #   virtual (0,0) -> physical (0,0), (2,2)
        #   virtual (0,1) -> physical (0,1), (2,3)
        #   virtual (1,0) -> physical (1,0), (3,2)
        #   virtual (1,1) -> physical (1,1), (3,3)
        
        # So for M1 term 1:
        #   A: virtual (0,0),(1,1) -> physical (0,0),(2,2),(1,1),(3,3)
        #   B: virtual (0,0),(1,1) -> physical (0,0),(2,2),(1,1),(3,3)
        #   C: virtual (0,0),(1,1) -> physical (0,0),(2,2),(1,1),(3,3)
        
        # This is getting very verbose. Given time constraints, let me use the
        # 56-term approach which is correct and simple.
        
        return result
    
    # Use the simpler 8-block Strassen approach: 8 * 7 = 56 terms for 4x4*4x4
    # Apply Strassen to each 2x2*2x2 block multiplication
    
    # C11 = A11*B11 + A12*B21
    terms.extend(strassen_2x2(0, 0, 0))  # A11*B11
    terms.extend(strassen_2x2(0, 2, 0))  # A12*B21
    
    # C12 = A11*B12 + A12*B22
    terms.extend(strassen_2x2(0, 0, 2))  # A11*B12
    terms.extend(strassen_2x2(0, 2, 2))  # A12*B22
    
    # C21 = A21*B11 + A22*B21
    terms.extend(strassen_2x2(2, 0, 0))  # A21*B11
    terms.extend(strassen_2x2(2, 2, 0))  # A22*B21
    
    # C22 = A21*B12 + A22*B22
    terms.extend(strassen_2x2(2, 0, 2))  # A21*B12
    terms.extend(strassen_2x2(2, 2, 2))  # A22*B22
    
    # For 4x4 * 4x3 (last 3 columns of B, k=4,5,6), use schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
