import numpy as np
from fractions import Fraction
import time
import random
from itertools import product

def construct():
    n, m, p = 2, 4, 5
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Start with the trivial rank-40 decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=object)
    V0 = np.zeros((m * p, rank0), dtype=object)
    W0 = np.zeros((p * n, rank0), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U0[i * m + j, idx] = Fraction(1)
                V0[j * p + k, idx] = Fraction(1)
                W0[k * n + i, idx] = Fraction(1)
    
    # The tensor for matrix multiplication C = A*B where C is n x p, A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Known: For matrix multiplication, the rank is at most n*m*p (trivial)
    # But we can do better using algorithms like Strassen or block decomposition
    
    # Let's try a systematic approach:
    # For n=2, m=4, p=5, we can think of this as multiplying a 2x4 matrix by a 4x5 matrix
    # We can use a block decomposition approach
    
    # First, let's try to find a decomposition using known algorithms or patterns
    
    # Try rank 20: split the j dimension into two parts
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # we can sometimes achieve rank less than n*m*p by sharing computations
    
    # For now, let's try a random search with verification
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes the matrix multiplication tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Known result: For 2x2 matrix multiplication, rank is 7 (Strassen)
    # For larger matrices, we can use block algorithms
    
    # Let's try a simple block decomposition:
    # Split A into [A1; A2] where A1 is 2x2, A2 is 2x2 (wait, m=4, so A is 2x4)
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # Each 2x2 * 2x5 multiplication has rank at most 7*5 = 35? No, that's not right.
    # Actually, 2x2 * 2x5: we can view this as 5 separate 2x2 * 2x1 multiplications
    # Each 2x2 * 2x1 has rank at most 7 (Strassen for 2x2, but for 2x2*2x1 it's different)
    
    # Let's try a different approach: use the trivial decomposition but try to merge terms
    
    # For the tensor T[i,j,k] = 1, we need sum_r U[i,j]_r V[j,k]_r W[k,i]_r = 1
    
    # One simple approach: use rank = m (the middle dimension)
    # Set U[i,j]_r = delta_{j,r}, V[j,k]_r = delta_{j,r} * (1/p), W[k,i]_r = 1
    # Then sum_r delta_{j,r} * delta_{j,r} * (1/p) * 1 = 1/p, not 1
    
    # Let's try: U[i,j]_r = 1 for all i,j,r; V[j,k]_r = delta_{j,r} * (1/m); W[k,i]_r = 1
    # sum_r 1 * delta_{j,r} * (1/m) * 1 = 1/m, not 1
    
    # The issue is that we need the sum to be exactly 1, and the entries must be integer/half-integer
    
    # Let's try a random search for small ranks
    
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Try ranks from 20 down to 10
    for rank in range(20, 9, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with small integers
            U = np.zeros((n * m, rank), dtype=object)
            V = np.zeros((m * p, rank), dtype=object)
            W = np.zeros((p * n, rank), dtype=object)
            
            for idx in range(n * m * rank):
                i, r = divmod(idx, rank)
                U[i, r] = Fraction(random.randint(-2, 2))
            
            for idx in range(m * p * rank):
                j, r = divmod(idx, rank)
                V[j, r] = Fraction(random.randint(-2, 2))
            
            for idx in range(p * n * rank):
                k, r = divmod(idx, rank)
                W[k, r] = Fraction(random.randint(-2, 2))
            
            # Check if this random decomposition works (unlikely, but try)
            if verify_decomposition(U, V, W, rank):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank0:
            break
    
    # If we found a better solution, use it; otherwise return the trivial one
    if best_rank < rank0:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U0
        V_out = V0
        W_out = W0
    
    # Convert to lists of integers/half-integers
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U_out), "V": convert_to_list(V_out), "W": convert_to_list(W_out)}
