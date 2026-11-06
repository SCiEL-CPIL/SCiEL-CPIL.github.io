import time
import numpy as np
from itertools import combinations

def _rref_gf2(M):
    """
    Compute RREF of M over GF(2).
    Returns:
    - rref: The RREF matrix (rank x n_cols)
    - pivot_cols: The indices of the pivot columns
    """
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    work = M.copy()
    rank = 0
    pivot_cols = []
    
    for col in range(n_cols):
        if rank >= n_rows:
            break
        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            
        # Eliminate
        for row in range(n_rows):
            if row != rank and work[row, col] == 1:
                work[row, :] ^= work[rank, :]
                
        pivot_cols.append(col)
        rank += 1
        
    rref = work[:rank, :]
    return rref, pivot_cols

def _synthesize_circuit(M, rref, pivot_cols):
    """
    Synthesize circuit for M using the given RREF and pivot columns.
    Returns gate count and the circuit dict.
    """
    n_rows, n_cols = M.shape
    rank = len(pivot_cols)
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_rows}

    next_wire = [n_cols + 1]
    gates = []
    wire_map = {}
    
    # Helper to build XOR tree with caching
    def build_xor_tree(indices):
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        key = tuple(sorted(indices))
        if key in wire_map:
            return wire_map[key]
        
        mid = len(indices) // 2
        left_w = build_xor_tree(indices[:mid])
        right_w = build_xor_tree(indices[mid:])
        
        gates.append([left_w, right_w])
        w_id = next_wire[0]
        next_wire[0] += 1
        wire_map[key] = w_id
        return w_id

    # Synthesize basis vectors (rows of RREF)
    basis_wires = []
    for i in range(rank):
        row = rref[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        w = build_xor_tree(ones)
        basis_wires.append(w)
        
    # Synthesize outputs
    outputs = []
    for j in range(n_rows):
        # Coefficients are M[j, pivot_cols]
        coeffs = M[j, pivot_cols]
        active = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        w = build_xor_tree(active)
        outputs.append(w)
        
    return len(gates), {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Compute RREF
    rref, pivot_cols = _rref_gf2(M)
    rank = len(pivot_cols)
    
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}

    # Strategy: 
    # 1. Start with the canonical RREF basis.
    # 2. Perform local search on the basis vectors to minimize total gate count.
    #    A basis change corresponds to multiplying the basis by an invertible matrix over GF(2).
    #    We can try flipping bits in the basis vectors (row additions) and see if it reduces cost.
    
    # First, get the initial cost
    initial_cost, initial_circuit = _synthesize_circuit(M, rref, pivot_cols)
    best_cost = initial_cost
    best_circuit = initial_circuit
    current_rref = rref.copy()
    
    # Local search: try adding one basis vector to another (row operation)
    # This changes the basis but keeps the row space the same.
    # We need to re-compute the "coefficients" of M in terms of the new basis.
    # If R is the basis matrix (rank x n_cols), then M = C @ R where C is (n_rows x rank).
    # Since R is in RREF, C is simply M[:, pivot_cols] if R is the standard RREF.
    # If we change the basis to R' = T @ R, then C' = C @ T^{-1}.
    # For a single row addition R_i <- R_i + R_j, T is identity with T[j,i] = 1.
    # T^{-1} is the same.
    # So C'_j = C_j + C_i for the j-th row of C? No.
    # Let's just recompute C by solving the system or by maintaining it.
    # Actually, it's easier to just compute the cost for the new basis directly.
    # The new basis vectors are the rows of R'.
    # The outputs are linear combinations of the new basis vectors.
    # To find the coefficients, we need to express M rows in terms of R' rows.
    # Since R' has the same column support structure roughly, we can use the pivot columns of R' if they are still independent.
    # However, a row operation R_i <- R_i + R_j might change the pivot structure if R_i and R_j have overlapping pivots?
    # No, RREF has unique pivots. R_i has pivot at col p_i. R_j has pivot at col p_j.
    # R_i + R_j has 1 at p_i and 1 at p_j. It is not in RREF form.
    # The "basis" is just a set of independent vectors. The synthesis cost depends on the Hamming weights of these vectors and the coefficients of the outputs.
    # We can define the cost function for any basis {v_0, ..., v_{rank-1}}.
    # We need to find the coefficients c_j such that M_j = sum c_{j,k} v_k.
    # This requires inverting the matrix formed by the basis vectors restricted to some independent set of columns.
    # Let's pick a set of 'rank' columns where the basis vectors are independent.
    # The original pivot_cols work for the original RREF.
    # After a row operation, are the new basis vectors still independent on the same pivot_cols?
    # Yes, because the transformation T is invertible. If R[:, pivot_cols] is invertible, then (T@R)[:, pivot_cols] = T @ (R[:, pivot_cols]) is also invertible.
    
    # So we can maintain the same pivot_cols and just update the coefficients C.
    # C = M[:, pivot_cols] @ (R[:, pivot_cols])^{-1}
    
    # Let's compute R_inv_pivot = (R[:, pivot_cols])^{-1}
    R_pivot = rref[:, pivot_cols]
    # Invert R_pivot over GF(2)
    # R_pivot is rank x rank identity matrix in standard RREF!
    # So C = M[:, pivot_cols] @ I = M[:, pivot_cols].
    
    # If we perform a row operation R_i <- R_i + R_j:
    # The new basis vectors are updated.
    # The new coefficient matrix C' satisfies M = C' @ R'.
    # C' = C @ T^{-1}.
    # T is identity with T[j,i] = 1 (since R'_i = R_i + R_j, so row i of T is e_i + e_j? No.
    # R' = T @ R.
    # R'_i = sum_k T[i,k] R_k.
    # If R'_i = R_i + R_j, then T[i,i]=1, T[i,j]=1, others 0 in row i.
    # So T is identity with T[i,j]=1.
    # T^{-1} = T (since T^2 = I).
    # So C' = C @ T.
    # C'_k = sum_l C_{k,l} T[l,k].
    # T[l,k] is 1 if l=k, plus 1 if l=i and k=j? No.
    # T is row i has 1 at col i and j.
    # So (C @ T)_{k, m} = sum_l C_{k,l} T[l,m].
    # T[l,m] is 1 if l=m, except for l=i where T[i,i]=1 and T[i,j]=1.
    # So (C @ T)_{k, i} = C_{k,i} + C_{k,j}
    # (C @ T)_{k, j} = C_{k,j}
    # (C @ T)_{k, m} = C_{k,m} for m != i, j
    
    # So the update to C is: C[:, i] ^= C[:, j].
    
    # Let's implement local search:
    # 1. Compute C = M[:, pivot_cols]
    # 2. Loop:
    #    a. For each pair (i, j), try R_i <- R_i + R_j.
    #    b. Update R and C.
    #    c. Compute cost.
    #    d. If cost < best_cost, accept. Else revert.
    #    e. Repeat until no improvement or time limit.
    
    C = M[:, pivot_cols].copy()
    current_R = rref.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(rank):
            if time.time() > deadline:
                break
            for j in range(rank):
                if i == j:
                    continue
                if time.time() > deadline:
                    break
                
                # Try R_i <- R_i + R_j
                new_R_i = current_R[i] ^ current_R[j]
                new_C_i = C[:, i] ^ C[:, j] # Wait, C update is C[:, i] ^= C[:, j]
                
                # Update current_R and C temporarily
                old_R_i = current_R[i].copy()
                old_C_i = C[:, i].copy()
                
                current_R[i] = new_R_i
                C[:, i] = new_C_i
                
                # Compute cost
                # We need to synthesize with basis current_R and coefficients C.
                # Note: The pivot columns are still valid for independence.
                cost, circuit = _synthesize_circuit_from_basis(M, current_R, C, pivot_cols)
                
                if cost < best_cost:
                    best_cost = cost
                    best_circuit = circuit
                    improved = True
                else:
                    # Revert
                    current_R[i] = old_R_i
                    C[:, i] = old_C_i
    
    return best_circuit

def _synthesize_circuit_from_basis(M, R, C, pivot_cols):
    """
    Synthesize circuit given basis vectors R (rank x n_cols) and coefficients C (n_rows x rank).
    pivot_cols is kept for consistency but not strictly needed if R and C are correct.
    """
    n_rows, n_cols = M.shape
    rank = R.shape[0]
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_rows}

    next_wire = [n_cols + 1]
    gates = []
    wire_map = {}
    
    # Helper to build XOR tree with caching
    def build_xor_tree(indices):
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        key = tuple(sorted(indices))
        if key in wire_map:
            return wire_map[key]
        
        mid = len(indices) // 2
        left_w = build_xor_tree(indices[:mid])
        right_w = build_xor_tree(indices[mid:])
        
        gates.append([left_w, right_w])
        w_id = next_wire[0]
        next_wire[0] += 1
        wire_map[key] = w_id
        return w_id

    # Synthesize basis vectors
    basis_wires = []
    for i in range(rank):
        row = R[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        w = build_xor_tree(ones)
        basis_wires.append(w)
        
    # Synthesize outputs
    outputs = []
    for j in range(n_rows):
        coeffs = C[j, :]
        active = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        w = build_xor_tree(active)
        outputs.append(w)
        
    return len(gates), {'gates': gates, 'outputs': outputs}
