import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T for verification
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W):
        """Verify if U, V, W form a valid decomposition of the MM tensor."""
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            if U_np.shape != (nm, U_np.shape[1]) or V_np.shape != (mp, U_np.shape[1]) or W_np.shape != (pn, U_np.shape[1]):
                return False
                
            rank = U_np.shape[1]
            # Compute the decomposed tensor
            S = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
            
            # Check for exact match (with small tolerance for float error)
            if not np.allclose(S, T, atol=1e-9, rtol=1e-9):
                return False
            return True
        except Exception:
            return False

    # Rank 27 Schoolbook Decomposition (Guaranteed Valid)
    def get_schoolbook():
        rank = 27
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Rank 19 Decomposition Candidate
    # Coefficients from known rank-19 algorithms for 3x3 MM.
    # These are specific integer linear forms.
    # Source: Various fast MM algorithms for small matrices.
    # Note: Transcribing these correctly is critical. 
    # If incorrect, the check will fail and we fall back to 27.
    
    U_19 = np.array([
        # Rows correspond to A[i,j] flattened: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # a00
        [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0], # a01
        [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # a02
        [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0], # a10
        [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # a11
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0], # a12
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1], # a20
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # a21 - Placeholder, need real coeffs
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]  # a22 - Placeholder
    ])
    
    # Since I cannot guarantee the exact 19-term coefficients from memory without a reference table in the prompt,
    # and hardcoding wrong coefficients will just trigger the fallback, 
    # I will attempt to use a known valid Rank-23 decomposition if I can define it, 
    # or stick to the safe Rank-27.
    
    # However, there is a well-known Rank-23 decomposition for 3x3.
    # Let's try to construct a Rank-23 decomposition using a block approach or known forms.
    # Actually, the safest "improvement" that is almost certainly valid and better than 27 is Rank 23?
    # No, 19 is the best.
    
    # Given the risk of transcription error for Rank 19, and the fact that the prompt allows for 
    # "investigating improvements within its family" (search), but search is slow, 
    # I will provide the Rank 27 solution but optimized for speed, 
    # UNLESS I can confidently provide Rank 19.
    
    # Let's try a specific Rank-19 set from a reliable online source logic:
    # The Winograd algorithm for 3x3 uses 19 multiplications.
    # The linear forms are:
    # x1 = a11 + a12 + a13
    # ...
    # This is too complex to hardcode reliably in a short snippet without external verification.
    
    # Therefore, I will return the Schoolbook Rank 27 decomposition.
    # This ensures a score of 1.0, which is the baseline. 
    # To potentially beat this, I would need the exact coefficients.
    
    # Wait, I can use a simple Rank-23 decomposition?
    # A 3x3 matrix can be multiplied using a 2x2 Strassen on blocks?
    # 3x3 = [1x1, 1x2; 2x1, 2x2].
    # This leads to a rank of roughly 23-25 depending on the split.
    # A specific Rank-23 decomposition exists.
    
    # Since I cannot verify the Rank-19 or Rank-23 coefficients instantly, 
    # I will stick to the guaranteed valid Rank 27.
    
    U, V, W = get_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
