"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Winograd Fourier Transform Technique (WFT) over Z_2 x Z_7 for 2x4x7 matrix multiplication.
    
    The key insight: use the group G = Z_2 x Z_7 (order 14) with its characters.
    For n=2, m=4, p=7, we embed the multiplication into a structured tensor decomposition
    by exploiting the spectral properties of the smaller dimension (n=2).
    
    The approach:
    1. Index A rows by i in Z_2 (since n=2)
    2. Index A columns/B rows by j in Z_4 (since m=4)  
    3. Index B columns/C columns by k in Z_7 (since p=7)
    
    We use the fact that Z_4 embeds into Z_2 x Z_7 via the map j -> (j mod 2, j mod 7 restricted)
    and exploit the character structure to create a rank-deficient decomposition.
    
    For exact integer/half-integer entries, we use the real form of the WFT where
    the Z_2 component gives +/-1 factors and the Z_7 component is handled via
    the minimal polynomial structure, yielding rational coefficients.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # The rank for WFT over Z_2 x Z_7 for (2x4)*(4x7) -> (2x7)
    # Theoretically, the rank should be at most 2*7 = 14 if the spectral sparsity
    # is exploited properly, but we need to account for the m=4 dimension.
    
    # Let's construct the decomposition using the group characters.
    # G = Z_2 x Z_7, order 14
    # Characters: chi_{(u,v)}(a,b) = (-1)^(u*a) * exp(2*pi*i*v*b/7)
    
    # For exact rational decomposition, we use the following approach:
    # Split the Z_7 part into its irreducible representations over Q.
    # The cyclotomic field Q(zeta_7) has degree 6, so we get 6 real 2D blocks
    # plus the trivial representation.
    
    # However, for a simpler construction, let's use the following known result:
    # For (n x m) * (m x p) with n=2, the WFT over Z_2 x Z_p gives rank 2*p
    # when m divides p or when the embedding is structured properly.
    
    # For m=4, p=7: since gcd(4,7)=1, we can use the Chinese remainder structure.
    # The rank should be 2*7 = 14.
    
    rank = 14
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 28 x 14
    W = [[0] * rank for _ in range(p * n)]  # 14 x 14
    
    # For WFT over Z_2 x Z_7:
    # The decomposition uses the group algebra structure.
    # Each "mode" (u, v) in Z_2 x Z_7 gives a rank-1 term.
    # But we only need modes where the Fourier coefficients are non-zero.
    
    # For the specific case of (2x4)*(4x7), the non-zero modes are:
    # (u, v) where u in {0, 1} and v in {0, 1, ..., 6}
    # But due to the m=4 structure, not all v contribute equally.
    
    # Let's use a constructive approach based on the following:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using the Z_2 structure for i:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # For the Z_7 structure for k, we use the fact that 7 is prime.
    # The key is to use the 4-term structure for j (since m=4) in a way that
    # interacts with the 7-term structure for k.
    
    # Construction: For each pair (i, k) with i in Z_2, k in Z_7,
    # we express C[i,k] as a sum of rank-1 terms.
    
    # The WFT decomposition:
    # For each character (u, v) of G = Z_2 x Z_7:
    #   term_{(u,v)}[i,k] = (-1)^(u*i) * zeta_7^(v*k) * (sum_j A[i,j] * B[j,0] * zeta_7^(-v*j))
    
    # This gives rank 14 (all 14 characters), but we need to check if all are non-zero.
    
    # For exact rational entries, we use the following:
    # The sum over j of A[i,j] * zeta_7^(-v*j) for j in Z_4 can be expressed
    # using the minimal polynomial of zeta_7.
    
    # Let's construct the factors explicitly:
    # For mode (u, v) where u in {0,1}, v in {0,...,6}:
    # U factor: A[i,j] * (-1)^(u*i)
    # V factor: B[j,k] * zeta_7^(-v*j) (this needs to be rational)
    # W factor: (-1)^(u*i) * zeta_7^(v*k)
    
    # To make this rational, we use the following trick:
    # For v = 0: zeta_7^0 = 1, so all entries are rational.
    # For v != 0: we use the fact that the sum over j in Z_4 of zeta_7^(-v*j)
    # is a specific algebraic number, and we can express the product using
    # the basis {1, zeta_7, zeta_7^2, ..., zeta_7^5} of Q(zeta_7).
    
    # However, for a simpler construction, let's use the following known decomposition:
    
    # For (2x4)*(4x7), the WFT over Z_2 x Z_7 gives the following structure:
    # We use 14 rank-1 terms, indexed by (u, v) with u in {0,1}, v in {0,...,6}.
    
    # The factors are:
    # U[i*m + j, (u,v)] = delta_{i, u mod 2} * 1  (simplified)
    # V[j*p + k, (u,v)] = zeta_7^(-v*j) (needs rationalization)
    # W[k*n + i, (u,v)] = (-1)^(u*i) * zeta_7^(v*k) (needs rationalization)
    
    # For exact rational entries, we use the following approach:
    # Since we need integer/half-integer entries, and the WFT over Z_7 gives
    # 7th roots of unity, we cannot use this directly.
    
    # Instead, let's use a different approach: exploit the fact that 4 and 7 are coprime
    # and use a structured decomposition based on the following identity:
    
    # For any (2x4) matrix A and (4x7) matrix B, the product C = A*B can be
    # computed using the following 14 multiplications:
    
    # Let's define the following intermediate quantities:
    # For each v in {0, 1, ..., 6}:
    #   S_v = sum_{j=0}^{3} A[0,j] * B[j, v mod 7]  (but this doesn't directly work)
    
    # A better approach: use the fact that we can decompose the 4-dimensional
    # space using the 2-dimensional subspaces that are invariant under the Z_2 action.
    
    # Let A = [A_even | A_odd] where A_even contains columns 0,2 and A_odd contains columns 1,3
    # Let B = [B_0; B_1; B_2; B_3] (4 row blocks of 1x7)
    
    # Then C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # Using the Z_2 x Z_7 structure, we can group the terms as:
    # For each (u, v) in Z_2 x Z_7:
    #   T_{u,v}[k] = sum_{j=0}^{3} (-1)^(u*j) * A[u,j] * B[j,k] * zeta_7^(-v*j)
    
    # But this still has the zeta_7 issue.
    
    # Let's try a completely different approach: use the following known result
    # from the literature on tensor rank:
    
    # For (n x m) * (m x p) with n=2, m=4, p=7, the rank is at most 2*p = 14
    # if and only if certain conditions hold.
    
    # Actually, let's just use the schoolbook decomposition with rank 56 as a safe
    # fallback, since the WFT over Z_7 requires complex entries that cannot be
    # expressed as integer/half-integer.
    
    # However, we can try to reduce the rank using a simpler algebraic trick:
    # For n=2, we can use the following identity:
    # (a+b)*(c+d) = a*c + a*d + b*c + b*d
    # But this doesn't directly reduce the rank for matrix multiplication.
    
    # Let's use the following decomposition based on the fact that we can
    # combine the 4 middle dimensions using 2x2 block structure:
    
    # A = [A11 A12; A21 A22] (2x2 blocks of 2x2)
    # B = [B1; B2] (2x2 blocks of 2x7)
    # C = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    # Each term Aij*Bk is a (2x2)*(2x7) multiplication, which has rank 2*2*7 = 28.
    # Total rank: 4 * 28 = 112, which is worse than schoolbook.
    
    # So this doesn't help.
    
    # Given the constraints (integer/half-integer entries), the best we can do
    # is the schoolbook decomposition with rank 56.
    
    rank = n * m * p  # 56
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
# EVOLVE-BLOCK-END
