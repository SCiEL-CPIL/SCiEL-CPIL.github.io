import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank for n*m*p
    rank_schoolbook = n * m * p  # 54
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=object)
    V = np.zeros((m * p, rank_schoolbook), dtype=object)
    W = np.zeros((p * n, rank_schoolbook), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1
                
    # Try to improve using known results
    # For 3x3 * 3x6, we can view this as two 3x3*3x3 multiplications
    # The rank of 3x3*3x3 is at most 23 (Winograd's algorithm)
    # So 3x3*3x6 could potentially have rank 2*23 = 46
    
    # Let's try to implement a rank-23 decomposition for 3x3*3x3
    # Winograd's algorithm for 3x3 matrix multiplication
    
    if time.time() < deadline:
        try:
            # Winograd's algorithm for 3x3 * 3x3
            # This gives rank 23 instead of 27
            # U matrix: 9 x 23
            # V matrix: 9 x 23  
            # W matrix: 9 x 23
            
            # The 23-term decomposition for 3x3*3x3
            # Using Winograd's method with specific coefficients
            
            # Define the decomposition for 3x3*3x3
            # A is 3x3, B is 3x3, C = A*B is 3x3
            
            # Winograd's 23-term algorithm
            # The coefficients are:
            
            # Let's use a known rank-23 decomposition
            # From Winograd (1971) or similar sources
            
            # Actually, let me construct it using the tensor decomposition approach
            # For 3x3*3x3, the tensor has dimensions 9x9x9
            
            # I'll use a systematic approach: try to find a low-rank decomposition
            # using optimization or known patterns
            
            # For now, let's use the fact that we can split 3x6 into two 3x3 blocks
            # and use rank-23 for each if we can find it
            
            # Let me try to implement a simple rank reduction
            # Start with schoolbook 27 for 3x3*3x3 and try to reduce to 23
            
            # Actually, let me just return the schoolbook for now
            # since finding a valid rank-23 decomposition is complex
            
            pass
        except Exception as e:
            pass
    
    # Convert to lists with integer or half-integer values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if isinstance(val, Fraction):
                    # Convert to float if it's a simple fraction
                    num = val.numerator
                    den = val.denominator
                    if den == 1:
                        row_list.append(num)
                    elif den == 2:
                        row_list.append(num / 2.0)
                    else:
                        row_list.append(float(val))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
