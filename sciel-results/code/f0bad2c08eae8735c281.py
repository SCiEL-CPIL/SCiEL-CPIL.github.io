"""Matrix multiplication 2x2 by 2x10: Strassen-based rank-35 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# Strassen rank-7 decomposition for 2x2 matrix multiplication
# M1 = (A11+A22)*(B11+B22)
# M2 = (A21+A22)*B11
# M3 = A11*(B12-B22)
# M4 = A22*(B21-B11)
# M5 = (A11+A12)*B22
# M6 = (A21-A11)*(B11+B12)
# M7 = (A12-A22)*(B21+B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

# We partition the 2x10 matrix B into five 2x2 blocks: B[:, 2b:2b+2] for b=0..4
# A is 2x2, each block is 2x2, product is 2x2
# Total: 5 independent 2x2 multiplications, each rank 7 -> total rank 35

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2x2
    
    # Strassen coefficients for 2x2 matmul
    # Input A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # Input B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # Output C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # Strassen 7 products with their A and B coefficient vectors
    # Each product: list of (a_idx, coeff) and (b_idx, coeff)
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    
    strassen_products = [
        # M1 = (A11+A22)*(B11+B22)
        {'a': [(0, 1), (3, 1)], 'b': [(0, 1), (3, 1)], 'c': [(0, 1), (3, -1)]},
        # M2 = (A21+A22)*B11
        {'a': [(2, 1), (3, 1)], 'b': [(0, 1)], 'c': [(2, 1), (3, -1)]},
        # M3 = A11*(B12-B22)
        {'a': [(0, 1)], 'b': [(1, 1), (3, -1)], 'c': [(1, 1), (3, 1)]},
        # M4 = A22*(B21-B11)
        {'a': [(3, 1)], 'b': [(2, 1), (0, -1)], 'c': [(0, 1), (2, 1)]},
        # M5 = (A11+A12)*B22
        {'a': [(0, 1), (1, 1)], 'b': [(3, 1)], 'c': [(0, -1), (1, 1)]},
        # M6 = (A21-A11)*(B11+B12)
        {'a': [(2, 1), (0, -1)], 'b': [(0, 1), (1, 1)], 'c': [(3, 1)]},
        # M7 = (A12-A22)*(B21+B22)
        {'a': [(1, 1), (3, -1)], 'b': [(2, 1), (3, 1)], 'c': [(0, 1)]},
    ]
    
    rank = 7 * num_blocks  # 35
    
    # U has shape (n*m, rank) = (4, 35)
    # V has shape (m*p, rank) = (20, 35)
    # W has shape (p*n, rank) = (20, 35)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            prod = strassen_products[s]
            
            # U: input A is 2x2, indexed as i*m+j = i*2+j
            # A index mapping: 0->A11(i=0,j=0), 1->A12(i=0,j=1), 2->A21(i=1,j=0), 3->A22(i=1,j=1)
            a_idx_to_row = {0: 0, 1: 1, 2: 2, 3: 3}
            for a_idx, coeff in prod['a']:
                U[a_idx_to_row[a_idx]][col] += coeff
            
            # V: input B is 2x10, indexed as j*p+k = j*10+k
            # Block b uses columns 2b and 2b+1
            # B index mapping within block: 0->B11(j=0,k=2b), 1->B12(j=0,k=2b+1), 2->B21(j=1,k=2b), 3->B22(j=1,k=2b+1)
            b_idx_to_row = {
                0: 0 * p + 2 * block,      # j=0, k=2b
                1: 0 * p + (2 * block + 1), # j=0, k=2b+1
                2: 1 * p + 2 * block,      # j=1, k=2b
                3: 1 * p + (2 * block + 1) # j=1, k=2b+1
            }
            for b_idx, coeff in prod['b']:
                V[b_idx_to_row[b_idx]][col] += coeff
            
            # W: output C is 2x2 per block, but the full output is 2x10
            # W index: k*n+i where k is column index in output, i is row index
            # For block b, output columns are 2b and 2b+1
            # C index mapping within block: 0->C11(i=0,k=2b), 1->C12(i=0,k=2b+1), 2->C21(i=1,k=2b), 3->C22(i=1,k=2b+1)
            c_idx_to_row = {
                0: (2 * block) * n + 0,       # k=2b, i=0
                1: (2 * block + 1) * n + 0,   # k=2b+1, i=0
                2: (2 * block) * n + 1,       # k=2b, i=1
                3: (2 * block + 1) * n + 1    # k=2b+1, i=1
            }
            for c_idx, coeff in prod['c']:
                W[c_idx_to_row[c_idx]][col] += coeff
    
    # Convert to lists (already are)
    return {"U": U, "V": V, "W": W}


def _verify():
    """Verify the decomposition is correct."""
    result = construct()
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i, j, k] = 1 for all valid (i,j,k)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # Actually the tensor is: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication (A * B), the (i,k) entry of result is sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    print(f"Verification: max error = {max_err}")
    return max_err < 1e-10


if __name__ == "__main__":
    _verify()
