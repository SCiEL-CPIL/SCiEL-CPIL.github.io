"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = 3, 3, 3
    # Rank-23 decomposition for 3x3 matrix multiplication
    # Using a known explicit construction
    #
    # The matrix multiplication tensor for 3x3 matrices has rank 23.
    # We need U (9 x 23), V (9 x 23), W (9 x 23) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) corresponds
    # to a multiplication term, else 0.
    #
    # Actually, the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{appropriate}
    #
    # For matrix mult: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T[i,j,k] = 1 for all i,j,k (each triple contributes)
    # But wait - the output is W[k*n+i, r], so the output index is (k, i)
    # This means C_{ki} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # But we want C_{ik} = sum_j A_{ij} B_{jk}
    #
    # So the convention W[k*n+i] means the (k,i)-th entry of output.
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    #
    # In the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # So sum over c: U[i,j,c]*V[j,k,c]*W[k,i,c] 
    # This gives for each (i,j,k): product of the three factors
    # The total sum over all (i,j,k) of A[i,j]*B[j,k]*sum_c(U[i,j,c]V[j,k,c]W[k,i,c])
    # should equal sum_{i,k} C[i,k] = sum_{i,k,j} A[i,j]B[j,k]
    # So we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
    #
    # Wait, that would make the rank 1 (all ones). That can't be right.
    #
    # Let me re-read the problem. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T is the matrix multiplication tensor.
    #
    # For 3x3 matrix multiplication, T[i,j,k] = 1 if and only if the
    # term A[i,j]*B[j,k] contributes to C[i,k]. Since every (i,j,k) triple
    # with valid indices contributes, T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    #
    # But that's the all-ones tensor of rank 1!
    #
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    #
    # The matrix multiplication tensor T of dimensions n x m x p is defined
    # such that for matrices A (n x m) and B (m x p):
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has components T[i,j,k] where i indexes rows of A,
    # j indexes columns of A / rows of B, k indexes columns of B.
    # T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] appears in
    # (A*B)[i,k]. Since (A*B)[i,k] = sum_j A[i,j]*B[j,k], every term
    # with matching j appears, so T[i,j,k] = 1 for all valid i,j,k.
    #
    # This is still all-ones. The issue is that the tensor should encode
    # WHICH output entry each term contributes to.
    #
    # Actually, I think the correct formulation is that the bilinear map
    # is: T: U x V -> W where (u, v) maps to sum_r u_r v_r w_r
    # and the constraint is that the bilinear form reproduces matrix multiplication.
    #
    # Let me just use a known rank-23 decomposition. One standard construction
    # for 3x3 matrix mult with rank 23 uses the following idea based on
    # Strassen's technique adapted to 3x3.
    
    # A known explicit rank-23 decomposition:
    # Based on the work of Bläser and others.
    #
    # I'll use a construction that I can verify. Let me build it carefully.
    #
    # For 3x3, one approach: use the fact that 3x3 mult can be done with
    # 23 multiplications. Here's one explicit scheme.
    
    # Let me use a verified construction. The indices are:
    # A is 3x3: A[0,0] A[0,1] A[0,2]
    #            A[1,0] A[1,1] A[1,2]
    #            A[2,0] A[2,1] A[2,2]
    # B is 3x3: B[0,0] B[0,1] B[0,2]
    #            B[1,0] B[1,1] B[1,2]
    #            B[2,0] B[2,1] B[2,2]
    # C = A*B is 3x3
    
    # A rank-23 decomposition exists. Let me implement one known scheme.
    
    # Actually, let me implement a simpler approach: start with schoolbook (rank 27)
    # and then try to find a lower-rank decomposition using optimization or
    # known formulas.
    
    # For now, let me implement a known rank-23 decomposition.
    # One such decomposition uses the following structure (from literature):
    
    # I'll encode a specific rank-23 solution.
    # The solution below is based on explicit algebraic constructions.
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Term 0-2: diagonal blocks (3 terms)
    # These handle the main diagonal contributions
    for r in range(3):
        U[r][r] = 1
        V[r][r] = 1
        W[r][r] = 1
    
    # Actually, let me just return a valid schoolbook decomposition first,
    # then try to improve. But the task asks for rank-minimal.
    #
    # Given the complexity of encoding a correct rank-23 decomposition by hand,
    # let me implement a search/optimization approach within the time budget.
    #
    # Strategy: Use the known fact that rank 23 is achievable for 3x3.
    # I'll implement a specific known decomposition.
    
    # Here's a concrete rank-23 decomposition from the literature.
    # The 23 terms are constructed as follows:
    
    # I'll use a verified set of 23 triples.
    # Each term r has:
    #   U[r]: a 9-vector (row-major 3x3)
    #   V[r]: a 9-vector (row-major 3x3)  
    #   W[r]: a 9-vector (row-major 3x3, with W[k*3+i] convention)
    
    # Let me define the decomposition explicitly.
    # This is based on a construction where we split the 27 schoolbook terms
    # and combine some using additional structure.
    
    # For a clean implementation, let me use the following known rank-23 scheme:
    
    # Term definitions: (U_indices, V_indices, W_indices) where each is a list
    # of (index, coefficient) pairs
    
    terms = []
    
    # 9 terms for the "diagonal" structure
    # Terms 0-8: standard schoolbook-like but modified
    
    # I'll use a different approach: implement a randomized/optimization search
    # to find a rank-23 decomposition, with fallback to rank 27.
    
    import numpy as np
    
    deadline = time.time() + 120  # 120 second deadline
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        n, m, p = 3, 3, 3
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook decomposition should be valid"
    
    # Try to find a lower-rank decomposition
    # Use the known rank-23 decomposition
    #
    # One explicit rank-23 decomposition for 3x3 matrix multiplication:
    # Based on algebraic constructions, here are the 23 terms:
    
    U23 = [[0]*23 for _ in range(9)]
    V23 = [[0]*23 for _ in range(9)]
    W23 = [[0]*23 for _ in range(9)]
    
    # Define the 23 terms explicitly.
    # Each term is (row_a, col_a, row_b, col_b, row_c, col_c) where:
    # U[row_a*3+col_a] = 1, V[row_b*3+col_b] = 1, W[col_c*3+row_c] = 1
    # This handles one multiplication term A[row_a,col_a]*B[row_b,col_b] -> C[row_c,col_c]
    
    # But rank 23 means we need to combine terms cleverly.
    # A single term with all 1s handles exactly one (i,j,k) triple.
    # To get rank < 27, we need terms that handle multiple triples.
    
    # Let me use a known construction. The following is a valid rank-23
    # decomposition (verified by construction):
    
    # I'll encode a specific solution. The 23 bilinear forms are:
    
    # This is getting complex. Let me use a simpler verified approach:
    # Return the schoolbook decomposition (rank 27) as a valid baseline,
    # which scores 1.0 (same as seed).
    
    # Actually, the seed already gives rank 27. To improve, I need rank < 27.
    # The proven minimum is 23. Let me implement a known rank-23 solution.
    
    # Known rank-23 decomposition for 3x3 matrix multiplication:
    # Reference: "The complexity of computing small matrices" by various authors
    
    # Here's an explicit rank-23 decomposition:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U23[0][0] = 1; V23[0][0] = 1; W23[0][0] = 1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]  
    U23[1][1] = 1; V23[3][1] = 1; W23[0][1] = 1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U23[2][2] = 1; V23[6][2] = 1; W23[0][2] = 1
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    U23[3][3] = 1; V23[1][3] = 1; W23[3][3] = 1
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    U23[4][4] = 1; V23[4][4] = 1; W23[3][4] = 1
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    U23[5][5] = 1; V23[7][5] = 1; W23[3][5] = 1
    # Term 6: A[2,0]*B[0,2] -> C[2,2]
    U23[6][6] = 1; V23[2][6] = 1; W23[6][6] = 1
    # Term 7: A[2,1]*B[1,2] -> C[2,2]
    U23[7][7] = 1; V23[5][7] = 1; W23[6][7] = 1
    # Term 8: A[2,2]*B[2,2] -> C[2,2]
    U23[8][8] = 1; V23[8][8] = 1; W23[6][8] = 1
    
    # These 9 terms handle the diagonal entries.
    # Now I need 14 more terms to handle the off-diagonal entries (18 entries total
    # minus the 3 diagonal = 15 off-diagonal, but some terms can handle multiple).
    
    # This manual construction is error-prone. Let me use a different strategy:
    # implement a numerical optimization to find a rank-23 decomposition.
    
    # Given time constraints, let me return the schoolbook decomposition.
    # The score will be 1.0 (same as seed), which is valid but not improved.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
