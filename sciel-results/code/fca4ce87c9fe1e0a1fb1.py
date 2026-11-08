import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_DEADLINE = time.time() + 115

def _strassen_222_terms():
    """Return list of 7 tuples (u_pairs, v_pairs, w_pairs) for Strassen 2x2x2."""
    # Each entry: ([(u_idx, coeff), ...], [(v_idx, coeff), ...], [(w_idx, coeff), ...])
    # Local indices: u is (i,j) with i in 0..1, j in 0..1; v is (j,k); w is (i,k)
    # We use flat local indices: u_ij = i*2+j, v_jk = j*2+k, w_ik = i*2+k
    return [
        # m1: (a00+a11)(b00+b11) -> c00, c11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # m2: (a10+a11)(b00) -> c10 - c11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # m3: a00(b01-b11) -> c01 + c11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # m4: a11(b10-b00) -> c00 + c10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # m5: (a00+a01)(b11) -> -c00 + c01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # m6: (a10-a00)(b00+b01) -> c11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # m7: (a01-a11)(b10+b11) -> c00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # We'll build U, V, W as numpy arrays for efficiency, then convert to lists
    # Start with schoolbook rank nm p = 72, but we'll use Strassen to reduce
    
    # Strategy: split p into 3 blocks of 2: p = 2+2+2
    # For each k-block of size 2:
    #   Split n into 1+2:
    #     n=1 block: M(1,4,2) - schoolbook rank 8
    #     n=2 block: M(2,4,2) - split m into 2+2, each M(2,2,2) rank 7, total 14
    #   Total per k-block: 8 + 14 = 22
    # Total: 3 * 22 = 66
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    def check_time():
        if time.time() - t0 > 100:
            return True
        return False
    
    def add_term(u_offsets, u_coeffs, v_offsets, v_coeffs, w_offsets, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for off, coeff in zip(u_offsets, u_coeffs):
            U[off, col] += coeff
        for off, coeff in zip(v_offsets, v_coeffs):
            V[off, col] += coeff
        for off, coeff in zip(w_offsets, w_coeffs):
            W[off, col] += coeff
        col += 1
    
    strassen_terms = _strassen_222_terms()
    
    # Process each k-block
    for kb in range(3):
        if check_time():
            break
        k_off = kb * 2  # offset in k dimension
        
        # n=0 (single row): M(1,4,2) schoolbook
        # u index: (0, j) = 0*4 + j = j
        # v index: (j, k) = j*6 + (k_off + k)
        # w index: (k, 0) = (k_off + k)*3 + 0
        for j in range(4):
            for k in range(2):
                u_idx = j  # (0, j) in (n*m) space
                v_idx = j * p + (k_off + k)
                w_idx = (k_off + k) * n + 0
                add_term([u_idx], [1], [v_idx], [1], [w_idx], [1])
        
        # n=1,2 (two rows): M(2,4,2)
        # Split m into 0,1 and 2,3
        for m_split in range(2):
            j_base = m_split * 2  # j offset for this 2-column block
            
            # Local mapping for M(2,2,2):
            # local u(i,j): i in 0..1, j in 0..1 -> global u = (1+i)*4 + (j_base+j)
            # local v(j,k): j in 0..1, k in 0..1 -> global v = (j_base+j)*6 + (k_off+k)
            # local w(i,k): i in 0..1, k in 0..1 -> global w = (k_off+k)*3 + (1+i)
            
            for u_pairs, v_pairs, w_pairs in strassen_terms:
                if check_time():
                    break
                u_offsets = [(1 + i) * m + (j_base + j) for (i, j), _ in 
                            [(idx // 2, idx % 2, c) for idx, c in u_pairs]]
                # Actually let me redo this more carefully
                u_offsets = []
                u_coeffs = []
                for idx, coeff in u_pairs:
                    i = idx // 2
                    j = idx % 2
                    u_offsets.append((1 + i) * m + (j_base + j))
                    u_coeffs.append(coeff)
                
                v_offsets = []
                v_coeffs = []
                for idx, coeff in v_pairs:
                    j = idx // 2
                    k = idx % 2
                    v_offsets.append((j_base + j) * p + (k_off + k))
                    v_coeffs.append(coeff)
                
                w_offsets = []
                w_coeffs = []
                for idx, coeff in w_pairs:
                    i = idx // 2
                    k = idx % 2
                    w_offsets.append((k_off + k) * n + (1 + i))
                    w_coeffs.append(coeff)
                
                add_term(u_offsets, u_coeffs, v_offsets, v_coeffs, w_offsets, w_coeffs)
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
