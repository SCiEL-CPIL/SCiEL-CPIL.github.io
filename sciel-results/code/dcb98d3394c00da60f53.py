"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    # Schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True
    
    def block_rank(a, b, c):
        """Return the rank needed for an a x b x c block."""
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c
    
    def get_partitions(total):
        """Generate all integer partitions of 'total' (order matters)."""
        results = []
        def gen(remaining, current):
            if remaining == 0:
                results.append(tuple(current))
                return
            for part in range(1, remaining + 1):
                current.append(part)
                gen(remaining - part, current)
                current.pop()
        gen(total, [])
        return results
    
    # Find optimal partition by enumerating all combinations
    n_parts_list = get_partitions(n)
    m_parts_list = get_partitions(m)
    p_parts_list = get_partitions(p)
    
    best_rank = n * m * p
    best_partition = (n_parts_list[0], m_parts_list[0], p_parts_list[0])
    
    for np_parts in n_parts_list:
        for mp_parts in m_parts_list:
            for pp_parts in p_parts_list:
                total_r = 0
                for a in np_parts:
                    for b in mp_parts:
                        for c in pp_parts:
                            total_r += block_rank(a, b, c)
                if total_r < best_rank:
                    best_rank = total_r
                    best_partition = (np_parts, mp_parts, pp_parts)
    
    # Convert partition sizes to (start, end) ranges
    def sizes_to_ranges(sizes):
        ranges = []
        start = 0
        for s in sizes:
            ranges.append((start, start + s))
            start += s
        return ranges
    
    n_ranges = sizes_to_ranges(best_partition[0])
    m_ranges = sizes_to_ranges(best_partition[1])
    p_ranges = sizes_to_ranges(best_partition[2])
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen(n_start, m_start, p_start):
        """Add Strassen decomposition for a 2x2x2 block starting at (n_start, m_start, p_start)."""
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # Term 1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # Term 2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # Term 3
            ([0, 0, 0, 1], [1, -1, 0, 0], [1, 1, 0, 0]),   # Term 4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # Term 5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # Term 6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # Term 7
        ]
        
        for u_coeff, v_coeff, w_coeff in terms:
            u_vec = np.zeros(n * m, dtype=np.int64)
            v_vec = np.zeros(m * p, dtype=np.int64)
            w_vec = np.zeros(p * n, dtype=np.int64)
            
            u_vec[n_start * m + m_start] = u_coeff[0]
            u_vec[n_start * m + m_start + 1] = u_coeff[1]
            u_vec[(n_start + 1) * m + m_start] = u_coeff[2]
            u_vec[(n_start + 1) * m + m_start + 1] = u_coeff[3]
            
            v_vec[m_start * p + p_start] = v_coeff[0]
            v_vec[m_start * p + p_start + 1] = v_coeff[1]
            v_vec[(m_start + 1) * p + p_start] = v_coeff[2]
            v_vec[(m_start + 1) * p + p_start + 1] = v_coeff[3]
            
            w_vec[p_start * n + n_start] = w_coeff[0]
            w_vec[p_start * n + n_start + 1] = w_coeff[1]
            w_vec[(p_start + 1) * n + n_start] = w_coeff[2]
            w_vec[(p_start + 1) * n + n_start + 1] = w_coeff[3]
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    def add_schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end):
        """Add schoolbook decomposition for a block."""
        for i in range(n_start, n_end):
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    u_vec = np.zeros(n * m, dtype=np.int64)
                    v_vec = np.zeros(m * p, dtype=np.int64)
                    w_vec = np.zeros(p * n, dtype=np.int64)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    U_list.append(u_vec)
                    V_list.append(v_vec)
                    W_list.append(w_vec)
    
    for n_start, n_end in n_ranges:
        for m_start, m_end in m_ranges:
            for p_start, p_end in p_ranges:
                n_size = n_end - n_start
                m_size = m_end - m_start
                p_size = p_end - p_start
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(n_start, m_start, p_start)
                else:
                    add_schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end)
    
    # Assemble the full matrices
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    if verify(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    else:
        # Fallback to schoolbook
        return schoolbook()
# EVOLVE-BLOCK-END
