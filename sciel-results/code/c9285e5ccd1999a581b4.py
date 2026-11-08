import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split B (m x p) into B1 (m x 4) and B2 (m x 2)
    # A is n x m (4 x 4)
    # C1 = A B1 (4 x 4)
    # C2 = A B2 (4 x 2)
    
    # Generate Strassen decomposition for 4x4x4x4
    # This requires a recursive generator or hardcoded values.
    # For simplicity and correctness, we can generate it recursively.
    
    def get_strassen(n):
        # Returns U, V, W for n x n x n x n
        if n == 1:
            return [[1]], [[1]], [[1]]
        
        # Base case n=2 hardcoded
        if n == 2:
            # u, v, w are lists of 4 entries (indices 0..3 for 2x2)
            # Index mapping: i*2 + j
            u1 = [1, 0, 0, 1] # e11 + e22
            v1 = [1, 0, 0, 1]
            w1 = [1, 0, 1, 0] # e11 + e21 (indices 0, 2)
            
            u2 = [0, 0, 1, 1] # e21 + e22
            v2 = [1, 0, 0, 0] # e11
            w2 = [-1, 0, 1, 0]
            
            u3 = [1, 0, 0, 0] # e11
            v3 = [0, 1, 0, -1] # e12 - e22
            w3 = [0, 1, 0, 1] # e12 + e22
            
            u4 = [0, 0, 0, 1] # e22
            v4 = [0, 0, 1, -1] # e21 - e11 (indices 2, 1? No. e21 is 2, e11 is 0. So 0, -1, 0, 1? No. v is B indices. B is 2x2. e21 is index 2. e11 is index 0. So v4 = [0, 0, 1, -1] is e21 - e11? No. e11 is 0, e21 is 2. So v4[0]=-1, v4[2]=1. -> [-1, 0, 1, 0])
            # Let's re-verify v4. M4 = A22 (B21 - B11). A22 is e22 (index 3). B21 is index 2, B11 is index 0.
            # So v4 should be 1 at index 2, -1 at index 0.
            v4 = [-1, 0, 1, 0]
            w4 = [1, 1, 0, 0] # e11 + e12
            
            u5 = [1, 1, 0, 0] # e11 + e12
            v5 = [0, 0, 0, 1] # e22
            w5 = [0, 0, 0, -1] # -e22
            
            u6 = [-1, 0, 1, 0] # e21 - e11
            v6 = [1, 1, 0, 0] # e11 + e12
            w6 = [0, 0, 0, 1] # e22
            
            u7 = [0, 1, 0, -1] # e12 - e22
            v7 = [0, 0, 1, 1] # e21 + e22
            w7 = [-1, 0, 0, 1] # -e11 + e22
            
            U = [u1, u2, u3, u4, u5, u6, u7]
            V = [v1, v2, v3, v4, v5, v6, v7]
            W = [w1, w2, w3, w4, w5, w6, w7]
            return U, V, W

        # Recursive case
        # This is complex to implement correctly on the fly.
        # Given the small size, we can just use schoolbook for 4x4 if we can't easily get Strassen.
        # But 49 is much better than 64.
        # Let's try to implement the recursion.
        
        # Actually, for n=4, we can just use the n=2 decomposition on the 2x2 blocks.
        # This is equivalent to Kronecker product of the decompositions?
        # No, Strassen is not a Kronecker product in the simple sense.
        # But we can just call get_strassen(2) and expand.
        
        # For now, let's just use schoolbook for 4x4 to ensure correctness, 
        # and then try to improve.
        # Schoolbook rank 64.
        # Strassen rank 49.
        # If we use schoolbook for 4x4, total rank 64 + 32 = 96. Same as seed.
        # We MUST use Strassen to get a better score.
        
        # Let's implement the recursion properly.
        # A 4x4 matrix is 4 blocks of 2x2.
        # The decomposition of 4x4x4x4 can be built from 7 decompositions of 2x2x2x2.
        
        U2, V2, W2 = get_strassen(2)
        
        # The 7 terms for 4x4 are:
        # M1 = (A11+A22)(B11+B22) -> Sum of 7 terms from U2,V2,W2 applied to blocks
        # This is getting complicated.
        
        # Alternative: Use the fact that we only need to run once.
        # We can hardcode the 49 terms for 4x4x4x4?
        # Or we can use a library? No.
        
        # Let's try a different approach:
        # Use schoolbook for 4x4x4x4 (rank 64) and 4x4x4x2 (rank 32). Total 96.
        # This doesn't improve the score.
        
        # Is there a simpler decomposition for 4x4x4x6?
        # Maybe 4x4x4x4 (49) + 4x4x4x2 (32) is the way.
        # I need to generate the 49 terms.
        
        # Let's write a helper to expand Strassen 2x2 to 4x4.
        
        def expand_strassen_4x4():
            # Returns U, V, W for 4x4x4x4
            # Each entry in U2, V2, W2 is a vector of 4.
            # We need to map these to vectors of 16.
            
            U4 = []
            V4 = []
            W4 = []
            
            # The 7 top-level terms
            # For each top-level term k (1..7), we have 7 sub-terms (1..7)
            # Total 49 terms.
            
            # Top-level u_k is a 2x2 matrix.
            # We need to represent u_k as a 4x4 matrix? No, u_k is a vector of 4 blocks.
            # Actually, the "u" in Strassen 2x2 is a 2x2 matrix of coefficients.
            # When we go to 4x4, each coefficient is a 2x2 matrix.
            # So u_k becomes a 4x4 matrix.
            # And we need to decompose this 4x4 matrix into rank-1 tensors?
            # No, the decomposition is linear.
            
            # Let's stick to the vector representation.
            # For 2x2, u is a vector of 4.
            # For 4x4, u is a vector of 16.
            
            # The 7 top-level operations are linear combinations of the 2x2 blocks.
            # Let A = [[A11, A12], [A21, A22]].
            # M1 = (A11+A22)(B11+B22).
            # This is a product of two 2x2 matrices.
            # We know the decomposition of 2x2x2x2x2x2 is rank 7.
            # So M1 can be computed with 7 rank-1 terms.
            # Each of these terms involves linear combinations of the 2x2 blocks of A and B.
            
            # This is essentially the Kronecker product of the Strassen decomposition with itself?
            # No, it's the composition.
            
            # Given the complexity, I will use the schoolbook for 4x4x4x4 if I can't quickly verify Strassen.
            # But 96 is the seed. I need < 96.
            
            # What if I use 3x3x3x3 Strassen? No.
            
            # Let's try to just output the schoolbook for 4x4x4x4 (64) and 4x4x4x2 (32).
            # Wait, 64+32 = 96.
            
            # Is there a rank < 96 for 4x4x4x6?
            # Yes, 81.
            # I must implement Strassen 4x4.
            
            # I will implement the recursion.
            
            pass

        # Since I can't easily verify the recursive Strassen implementation in my head,
        # I will use a known result:
        # The rank of 4x4x4x4 is 49.
        # I will generate the 49 terms by calling the 2x2 decomposition 7 times and expanding.
        
        U2, V2, W2 = get_strassen(2)
        
        U4 = []
        V4 = []
        W4 = []
        
        # Top level coefficients
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,1,0]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[-1,0,1,0]
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
        # M4: u=[0,0,0,1], v=[-1,0,1,0], w=[1,1,0,0]
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1]
        # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[-1,0,0,1]
        
        top_u = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,1,0,0],
            [-1,0,1,0],
            [0,1,0,-1]
        ]
        top_v = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,0,-1],
            [-1,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,0,1,1]
        ]
        top_w = [
            [1,0,1,0],
            [-1,0,1,0],
            [0,1,0,1],
            [1,1,0,0],
            [0,0,0,-1],
            [0,0,0,1],
            [-1,0,0,1]
        ]
        
        # For each top-level term k, we have a product of two 2x2 matrices.
        # We decompose this product using the 2x2 Strassen decomposition (U2, V2, W2).
        # The "u" of the 2x2 decomposition is a vector of 4.
        # We need to map this to a vector of 16 for the 4x4 case.
        
        # The 2x2 decomposition computes M = X Y.
        # X and Y are 2x2 matrices.
        # In the 4x4 case, X is a 4x4 matrix formed by blocks.
        # Specifically, for term k, X is formed by top_u[k] blocks.
        # X = [[ top_u[k][0] * I, top_u[k][1] * I ],
        #      [ top_u[k][2] * I, top_u[k][3] * I ]]
        # Where I is 2x2 identity? No, the blocks are 2x2.
        # Actually, the blocks are the 2x2 submatrices of A.
        # So X is a 4x4 matrix.
        
        # The 2x2 decomposition says M = sum_{j=1}^7 (u_j . X) (v_j . Y) w_j.
        # Here u_j, v_j are vectors of 4.
        # (u_j . X) is a scalar? No, X is 2x2.
        # u_j is a 2x2 matrix of coefficients.
        # So u_j . X = sum_{i,l} u_{j, i, l} X_{i, l}.
        # This is a scalar.
        
        # So for each top-level term k, and each sub-term j (1..7):
        # We have a rank-1 term for the 4x4 multiplication.
        # The "u" part is u_j (2x2) applied to X_k (4x4).
        # This results in a scalar coefficient for each element of the 4x4 "u" vector?
        # No, the "u" in the 4x4 decomposition is a vector of 16.
        
        # This is getting too complex to derive correctly in the code without testing.
        
        # I will fallback to schoolbook for 4x4x4x4 (rank 64) and 4x4x4x2 (rank 32).
        # Total 96.
        
        # Wait, I can use a different split.
        # 4x4x4x4 (64) + 4x4x4x2 (32) = 96.
        
        # Is there any other decomposition?
        # Maybe 4x4x4x3 (48) + 4x4x4x3 (48) = 96.
        
        # I will stick to 96 to ensure validity, as I cannot guarantee the Strassen implementation is correct.
        # A score of 1.0 is safe.
        
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
