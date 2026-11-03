"""Improved program for matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Total Rank calculation:
    # Partition I={0,1,2,3} into I0={0,1}, I1={2,3}
    # Partition J={0,1,2,3,4} into J0={0,1}, J1={2,3,4}
    # Partition K={0,1,2,3,4} into K0={0,1}, K1={2,3,4}
    
    # Blocks:
    # (2,2,2) -> Rank 7
    # (2,2,3) -> Rank 11 (7 for 2x2x2 part + 4 for 2x2x1 part)
    # (2,3,2) -> Rank 11 (7 for 2x2x2 part + 4 for 2x1x2 part)
    # (2,3,3) -> Rank 17 (7 for 2x2x2 + 4 for 2x2x1 + 4 for 2x1x2 + 2 for 2x1x1)
    
    # Total Rank = 2 * (7 + 11 + 11 + 17) = 92
    
    total_rank = 92
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    def build_strassen_2x2x2(offset_i, offset_j, offset_k, U, V, W, rank_start):
        """
        Builds Strassen decomposition for 2x2x2 block.
        Returns next rank index.
        """
        r = rank_start
        # Coefficients for U (4x7), V (4x7), W (4x7)
        # Local indices i,j,k in 0..1
        # Global indices gi = offset_i + i, gj = offset_j + j, gk = offset_k + k
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        U[offset_i*2 + 0, r] = 1; U[(offset_i+1)*2 + 1, r] = 1
        V[offset_j*2 + 0, r] = 1; V[(offset_j+1)*2 + 1, r] = 1
        W[offset_k*2 + 0, r] = 1; W[(offset_k+1)*2 + 1, r] = 1
        r += 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (1,0)=1, (1,1)=-1
        U[(offset_i+1)*2 + 0, r] = 1; U[(offset_i+1)*2 + 1, r] = 1
        V[offset_j*2 + 0, r] = 1
        W[(offset_k+1)*2 + 0, r] = 1; W[(offset_k+1)*2 + 1, r] = -1
        r += 1
        
        # Term 3: M3 = A00(B01-B11)
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: (0,1)=1, (1,1)=1
        U[offset_i*2 + 0, r] = 1
        V[offset_j*2 + 1, r] = 1; V[(offset_j+1)*2 + 1, r] = -1
        W[offset_k*2 + 1, r] = 1; W[(offset_k+1)*2 + 1, r] = 1
        r += 1
        
        # Term 4: M4 = A11(B10-B00)
        # U: (1,1)=1
        # V: (1,0)=1, (0,0)=-1
        # W: (0,0)=1, (1,0)=1
        U[(offset_i+1)*2 + 1, r] = 1
        V[(offset_j+1)*2 + 0, r] = 1; V[offset_j*2 + 0, r] = -1
        W[offset_k*2 + 0, r] = 1; W[(offset_k+1)*2 + 0, r] = 1
        r += 1
        
        # Term 5: M5 = (A00+A01)B11
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: (0,0)=-1, (0,1)=1
        U[offset_i*2 + 0, r] = 1; U[offset_i*2 + 1, r] = 1
        V[(offset_j+1)*2 + 1, r] = 1
        W[offset_k*2 + 0, r] = -1; W[offset_k*2 + 1, r] = 1
        r += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U: (1,0)=1, (0,0)=-1
        # V: (0,0)=1, (0,1)=1
        # W: (1,1)=1
        U[(offset_i+1)*2 + 0, r] = 1; U[offset_i*2 + 0, r] = -1
        V[offset_j*2 + 0, r] = 1; V[offset_j*2 + 1, r] = 1
        W[(offset_k+1)*2 + 1, r] = 1
        r += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # U: (0,1)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: (0,0)=1
        U[offset_i*2 + 1, r] = 1; U[(offset_i+1)*2 + 1, r] = -1
        V[(offset_j+1)*2 + 0, r] = 1; V[(offset_j+1)*2 + 1, r] = 1
        W[offset_k*2 + 0, r] = 1
        r += 1
        
        return r

    def build_schoolbook_block(n_b, m_b, p_b, offset_i, offset_j, offset_k, U, V, W, rank_start):
        """
        Builds a schoolbook block for dimensions n_b x m_b x p_b.
        """
        r = rank_start
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    gi = offset_i + i
                    gj = offset_j + j
                    gk = offset_k + k
                    U[gi * m + gj, r] = 1
                    V[gj * p + gk, r] = 1
                    W[gk * n + gi, r] = 1
                    r += 1
        return r

    def build_2x2x3_block(offset_i, offset_j, offset_k, U, V, W, rank_start):
        """
        Builds 2x2x3 block using Strassen for 2x2x2 part and Schoolbook for 2x2x1 part.
        J and K are split.
        Here, the '3' dimension is K.
        K is split into K0 (size 2) and K1 (size 1).
        """
        # Strassen for 2x2x2 (K0)
        rank_start = build_strassen_2x2x2(offset_i, offset_j, offset_k, U, V, W, rank_start)
        # Schoolbook for 2x2x1 (K1)
        # K1 starts at offset_k + 2
        rank_start = build_schoolbook_block(2, 2, 1, offset_i, offset_j, offset_k + 2, U, V, W, rank_start)
        return rank_start

    def build_2x3x2_block(offset_i, offset_j, offset_k, U, V, W, rank_start):
        """
        Builds 2x3x2 block using Strassen for 2x2x2 part and Schoolbook for 2x1x2 part.
        J is split into J0 (size 2) and J1 (size 1).
        """
        # Strassen for 2x2x2 (J0)
        rank_start = build_strassen_2x2x2(offset_i, offset_j, offset_k, U, V, W, rank_start)
        # Schoolbook for 2x1x2 (J1)
        # J1 starts at offset_j + 2
        rank_start = build_schoolbook_block(2, 1, 2, offset_i, offset_j + 2, offset_k, U, V, W, rank_start)
        return rank_start

    def build_2x3x3_block(offset_i, offset_j, offset_k, U, V, W, rank_start):
        """
        Builds 2x3x3 block.
        J split into J0 (2), J1 (1).
        K split into K0 (2), K1 (1).
        Blocks:
        1. 2x2x2 (J0, K0) -> Strassen (7)
        2. 2x2x1 (J0, K1) -> Schoolbook (4)
        3. 2x1x2 (J1, K0) -> Schoolbook (4)
        4. 2x1x1 (J1, K1) -> Schoolbook (2)
        """
        # 1. Strassen for 2x2x2 (J0, K0)
        rank_start = build_strassen_2x2x2(offset_i, offset_j, offset_k, U, V, W, rank_start)
        # 2. Schoolbook for 2x2x1 (J0, K1)
        rank_start = build_schoolbook_block(2, 2, 1, offset_i, offset_j, offset_k + 2, U, V, W, rank_start)
        # 3. Schoolbook for 2x1x2 (J1, K0)
        rank_start = build_schoolbook_block(2, 1, 2, offset_i, offset_j + 2, offset_k, U, V, W, rank_start)
        # 4. Schoolbook for 2x1x1 (J1, K1)
        rank_start = build_schoolbook_block(2, 1, 1, offset_i, offset_j + 2, offset_k + 2, U, V, W, rank_start)
        return rank_start

    rank_idx = 0
    
    # I0 = {0,1}, I1 = {2,3}
    # J0 = {0,1}, J1 = {2,3,4}
    # K0 = {0,1}, K1 = {2,3,4}
    
    # Block 1: I0, J0, K0 (2x2x2)
    rank_idx = build_strassen_2x2x2(0, 0, 0, U, V, W, rank_idx)
    
    # Block 2: I0, J0, K1 (2x2x3)
    rank_idx = build_2x2x3_block(0, 0, 2, U, V, W, rank_idx)
    
    # Block 3: I0, J1, K0 (2x3x2)
    rank_idx = build_2x3x2_block(0, 2, 0, U, V, W, rank_idx)
    
    # Block 4: I0, J1, K1 (2x3x3)
    rank_idx = build_2x3x3_block(0, 2, 2, U, V, W, rank_idx)
    
    # Block 5: I1, J0, K0 (2x2x2)
    rank_idx = build_strassen_2x2x2(2, 0, 0, U, V, W, rank_idx)
    
    # Block 6: I1, J0, K1 (2x2x3)
    rank_idx = build_2x2x3_block(2, 0, 2, U, V, W, rank_idx)
    
    # Block 7: I1, J1, K0 (2x3x2)
    rank_idx = build_2x3x2_block(2, 2, 0, U, V, W, rank_idx)
    
    # Block 8: I1, J1, K1 (2x3x3)
    rank_idx = build_2x3x3_block(2, 2, 2, U, V, W, rank_idx)
    
    if rank_idx != total_rank:
        print(f"Warning: Rank index {rank_idx} does not match expected {total_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
