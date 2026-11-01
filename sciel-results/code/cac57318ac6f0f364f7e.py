"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # m=4=2*2, p=7 is prime. Use WFT on the inner product of length m=4.
    # For each pair (i, k) we need to compute sum_{j=0}^{3} A[i,j] * B[j,k].
    # We pad B to length 4 and use a 4-point transform.
    # The standard 4-point WFT uses the factors from the factorization 4=2*2.
    # However, for small sizes, the "rank" of the inner product convolution is typically m.
    # But WFT can sometimes reduce this if there are symmetries or if we exploit the structure of the full tensor.
    # For matrix multiplication M_{n,m,p}, the tensor is a sum over j.
    # The tensor is T = sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k.
    # We can view this as for each (i,k), the vector of length m is multiplied.
    # This is a direct sum of n*p instances of the "1xmx1" multiplication (inner product).
    # The rank of the 1xmx1 multiplication (convolution of length m) is m for general m, but can be lower for specific m.
    # For m=4, the rank of the 4-point inner product is 4? 
    # Actually, the rank of the tensor for multiplying a 1x1x4 matrix? No.
    # The rank of the 4-point convolution (cyclic) is 4.
    # Wait, the rank of the 4-point cyclic convolution tensor is 4?
    # For 2x2, it's 3 (Strassen).
    # For 4x4, if we use Strassen recursively, rank is 49? No, 4x4 multiplication rank is 49 (schoolbook) or less?
    # Strassen for 4x4 uses 7 multiplications of 2x2, each rank 3, so 7*3=21? No, Strassen for 2x2 has rank 7.
    # So 4x4 via Strassen has rank 7 * 4 = 28? No.
    # Strassen: 2x2 uses 7 scalars. 4x4 uses 7 blocks of 2x2. Each 2x2 block uses 7 scalars. Total 49? No.
    # The number of multiplications is 7 * (n/2)^d. For 4x4, 7*2^2 = 28 multiplications.
    # But here we are multiplying 1x4 by 4x1 (inner product).
    # The rank of the 4-point inner product tensor is 4? 
    # Let's check the rank of the tensor T_4 = sum_{j=0}^3 e_j \otimes e_j \otimes 1.
    # This is just 4 terms. Can it be done with fewer?
    # No, the rank of the 4-point identity tensor (diagonal) is 4.
    # Wait, the tensor for inner product <a,b> is sum_j a_j b_j.
    # This is the 4-dimensional identity tensor in the second slot?
    # No, it's a vector of rank 1 in the output.
    # The tensor is T = sum_{j=0}^{3} e_j^{(A)} \otimes e_j^{(B)} \otimes 1^{(C)}.
    # This tensor has rank 4. You cannot decompose sum e_j \otimes e_j \otimes 1 into fewer than 4 terms because the supports on A and B are disjoint in the "diagonal" sense? 
    # Actually, the rank of the identity tensor I_n (sum e_i \otimes e_i) is n.
    # So the rank of the 4-point inner product is 4.
    # Since we have n*p = 2*7 = 14 such independent inner products, the total rank is 14 * 4 = 56.
    # This is the same as schoolbook.
    # WFT does not reduce the rank for the inner product of length 4 below 4.
    # However, if we consider the full 2x4x7 tensor, maybe there is global structure?
    # The tensor is T = sum_{i,k} ( sum_{j} A_{i,j} B_{j,k} ) e_{i,k}^{(C)}? No, C is 14x1? No, C is 2x7.
    # The output is a 2x7 matrix.
    # The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    # This is a sum of 14 independent rank-4 tensors (one for each i,k).
    # Since the supports on A and B for different (i,k) are disjoint (A_{i,j} only appears for that i, B_{j,k} only for that k), the total rank is the sum of the ranks.
    # Rank = sum_{i,k} Rank(T_{i,k}) = 14 * 4 = 56.
    # So WFT does not help here to reduce rank below 56 for this specific dimension.
    # We return the schoolbook decomposition which is rank 56.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
