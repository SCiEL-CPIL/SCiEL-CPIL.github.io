"""Matrix multiplication 3,3,7 with rank-minimal decomposition."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a block decomposition approach:
    - Split the 3x7 matrix B into blocks: B1 (3x3), B2 (3x3), B3 (3x1)
    - Use schoolbook multiplication for each block
    - Total rank = 9 + 9 + 3 = 21
    
    Then we attempt to reduce the rank further using optimization.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 63)
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook_decomposition()
    
    # Try block decomposition: split B into 3x3, 3x3, 3x1 blocks
    def block_decomposition():
        """
        Split the 3x7 matrix into blocks and use schoolbook for each.
        Block 1: A (3x3) * B1 (3x3) -> C1 (3x3), rank 9
        Block 2: A (3x3) * B2 (3x3) -> C2 (3x3), rank 9  
        Block 3: A (3x3) * B3 (3x1) -> C3 (3x1), rank 3
        Total rank: 21
        """
        total_rank = 9 + 9 + 3
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        
        # Block 1: columns 0-2 of B (3x3 block)
        for i in range(n):
            for j in range(m):
                for k in range(3):  # k = 0, 1, 2
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        # Block 2: columns 3-5 of B (3x3 block)
        for i in range(n):
            for j in range(m):
                for k in range(3, 6):  # k = 3, 4, 5
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        # Block 3: column 6 of B (3x1 block)
        for i in range(n):
            for j in range(m):
                k = 6
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W, total_rank
    
    U_block, V_block, W_block, rank_block = block_decomposition()
    
    # Use the block decomposition as our starting point
    U, V, W = U_block, V_block, W_block
    rank = rank_block
    
    # Verify the decomposition is correct using fast einsum
    def verify_decomposition(U, V, W):
        """
        Verify that the tensor decomposition correctly represents matrix multiplication.
        
        For rank-r decomposition, we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        """
        # Reshape U, V, W to 3D tensors
        U_3d = U.reshape(n, m, -1)  # (n, m, rank)
        V_3d = V.reshape(m, p, -1)  # (m, p, rank)
        W_3d = W.reshape(p, n, -1)  # (p, n, rank)
        
        # Compute the decomposed tensor using einsum
        # decomposed[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        decomposed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        target = np.ones((n, m, p))
        
        return np.allclose(decomposed, target, atol=1e-10)
    
    if not verify_decomposition(U, V, W):
        # Fall back to schoolbook
        U, V, W, rank = schoolbook_decomposition()
    
    # Now try to reduce the rank using iterative refinement
    # We'll try to remove linearly dependent columns
    
    # Check for linear dependence in the U, V, W matrices
    def remove_dependent_columns(U, V, W, max_rank):
        """
        Try to reduce rank by checking if any column is linearly dependent.
        This is a heuristic - we check small subsets.
        """
        current_rank = U.shape[1]
        if current_rank <= max_rank:
            return U, V, W, current_rank
        
        # Try removing one column at a time
        for col_to_remove in range(current_rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Temporarily remove this column
            U_trial = np.delete(U, col_to_remove, axis=1)
            V_trial = np.delete(V, col_to_remove, axis=1)
            W_trial = np.delete(W, col_to_remove, axis=1)
            
            # Check if the decomposition still holds
            if verify_decomposition(U_trial, V_trial, W_trial):
                U, V, W = U_trial, V_trial, W_trial
                current_rank = U.shape[1]
                if current_rank <= max_rank:
                    break
        
        return U, V, W, current_rank
    
    # Try to reduce rank
    U, V, W, rank = remove_dependent_columns(U, V, W, rank)
    
    # Convert to lists for the output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
