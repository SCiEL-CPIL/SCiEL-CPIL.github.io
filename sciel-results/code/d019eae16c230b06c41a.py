import time
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin for 60s limit
    
    if not matrix or not matrix[0]:
        return {'gates': [], 'outputs': []}
        
    m = len(matrix)
    n = len(matrix[0])
    
    # Convert to int bitmasks
    # Bit j corresponds to column j
    orig_rows = []
    for r in matrix:
        val = 0
        for j in range(n):
            if r[j]:
                val |= (1 << j)
        orig_rows.append(val)
        
    # A is the working matrix (list of ints)
    A = orig_rows[:]
    
    # used_pivots[i] is True if row i of A has been selected as a pivot
    used_pivots = [False] * m
    # pivot_cols[col] = row_idx in A that is the pivot for this column
    # If no pivot, -1
    pivot_cols = [-1] * n
    
    # We iterate through columns
    # To minimize cost, we look ahead? 
    # Standard GE goes left to right. 
    # The choice of pivot in column c affects the matrix for columns > c.
    
    for col in range(n):
        if time.time() > deadline:
            break # Return whatever we have (might be invalid if not full rank handled, but we'll handle at end)
            
        # Find candidates: rows i where A[i] has bit col set, and not used as pivot
        candidates = []
        for i in range(m):
            if not used_pivots[i] and (A[i] >> col) & 1:
                candidates.append(i)
                
        if not candidates:
            continue
            
        best_cand = -1
        best_cost = float('inf')
        
        # Simulate each candidate
        for i in candidates:
            # Copy A
            new_A = A[:]
            pivot_row = new_A[i]
            
            # Eliminate col from all other rows
            # We eliminate from ALL rows to get RREF
            for j in range(m):
                if j != i and (new_A[j] >> col) & 1:
                    new_A[j] = new_A[j] ^ pivot_row
            
            # Calculate Cost of new_A (weight of basis)
            w_A = sum(bin(row).count('1') for row in new_A)
            
            # Calculate Cost of C (coefficients)
            # The pivot columns are the current ones + this new one
            # We need to count 1s in orig_rows at all pivot column positions
            # Current pivots:
            # 1. Existing pivots from previous columns
            # 2. This new pivot (col, i)
            
            # Let's maintain a list of active pivot columns
            # active_pivots is a list of cols that have been assigned pivots
            # In the simulation, we assume we pick i for col.
            
            # We need to know which cols are pivots.
            # We can just re-calculate from scratch in the simulation?
            # Or maintain a set.
            
            # For simplicity in simulation:
            # The set of pivot columns will be {c' | c' < col and pivot_cols[c'] != -1} U {col}
            
            pivot_col_set = set()
            for c in range(col):
                if pivot_cols[c] != -1:
                    pivot_col_set.add(c)
            pivot_col_set.add(col)
            
            w_C = 0
            for r in orig_rows:
                # Count 1s in r at positions in pivot_col_set
                mask = 0
                for c in pivot_col_set:
                    if (r >> c) & 1:
                        mask |= (1 << c)
                w_C += bin(mask).count('1')
                
            total_cost = w_A + w_C
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_cand = i
                
        if best_cand != -1:
            # Apply the best choice
            pivot_row = A[best_cand]
            for j in range(m):
                if j != best_cand and (A[j] >> col) & 1:
                    A[j] = A[j] ^ pivot_row
            used_pivots[best_cand] = True
            pivot_cols[col] = best_cand
            
    # At this point, A is RREF (or as close as we got)
    # We need to build the circuit
    
    # Identify basis rows: non-zero rows in A that are pivots?
    # Actually, any non-zero row in RREF is a basis vector.
    # The pivot rows are the ones selected.
    # Let's collect the basis vectors and their indices in A.
    
    basis_indices = []
    for i in range(m):
        if A[i] != 0:
            # Check if it's a pivot row?
            # In RREF, every non-zero row has a pivot.
            # We marked used_pivots.
            # Is it possible to have a non-zero row that wasn't marked?
            # If we skipped a column, we didn't mark a pivot.
            # But if A[i] != 0, it must have a leading 1.
            # If that leading 1 was in a column where we didn't pick a pivot (because no candidate? impossible if A[i]!=0),
            # or we picked a different row.
            # Actually, if A[i] != 0, it corresponds to a pivot row in the final RREF.
            # My loop marks `used_pivots` when a pivot is selected.
            # So `used_pivots[i]` should be True for all non-zero rows in A?
            # Yes, because we only stop processing a column if no candidates.
            # If a row is non-zero, it has a leading 1. That column must have had a candidate (the row itself).
            # So it would have been picked or eliminated?
            # If it was eliminated, it would be 0.
            # So yes, non-zero rows in A are exactly the pivot rows.
            
            basis_indices.append(i)
            
    # Map basis row index in A to a wire ID
    # We need to compute these wires from inputs.
    
    gates = []
    wire_id_counter = n # 0..n-1 are inputs, n is 0
    
    def get_new_wire():
        nonlocal wire_id_counter
        w = wire_id_counter
        wire_id_counter += 1
        return w

    def xor_gate(w1, w2):
        # Check if w1 or w2 is 0
        if w1 == n: return w2
        if w2 == n: return w1
        if w1 == w2: return n # XOR with self is 0
        gates.append([w1, w2])
        return get_new_wire()

    def xor_list(wires):
        if not wires: return n
        if len(wires) == 1: return wires[0]
        
        # Balanced reduction
        next_wires = []
        it = iter(wires)
        for a in it:
            try:
                b = next(it)
                next_wires.append(xor_gate(a, b))
            except StopIteration:
                next_wires.append(a)
        
        while len(next_wires) > 1:
            wires = next_wires
            next_wires = []
            it = iter(wires)
            for a in it:
                try:
                    b = next(it)
                    next_wires.append(xor_gate(a, b))
                except StopIteration:
                    next_wires.append(a)
        
        return next_wires[0]

    basis_wires = {}
    
    # 1. Compute Basis Wires
    for i in basis_indices:
        row_mask = A[i]
        # Get input indices
        inputs = []
        for j in range(n):
            if (row_mask >> j) & 1:
                inputs.append(j)
                
        if not inputs:
            basis_wires[i] = n
        else:
            basis_wires[i] = xor_list(inputs)
            
    # 2. Compute Output Wires
    outputs = []
    
    # We need the pivot columns to determine coefficients
    # pivot_cols[c] = row_idx in A
    # For each original row r, coeff for basis row i is 1 iff r has 1 at pivot_col for i.
    
    # Create a mapping: basis_row_idx -> pivot_col
    basis_to_pivot_col = {}
    for c in range(n):
        r_idx = pivot_cols[c]
        if r_idx != -1:
            basis_to_pivot_col[r_idx] = c
            
    for r_idx in basis_indices:
        if r_idx not in basis_to_pivot_col:
             # Should not happen
             basis_to_pivot_col[r_idx] = -1

    for i in range(m):
        orig_row = orig_rows[i]
        
        # Determine which basis vectors are needed
        basis_indices_needed = []
        for b_idx, p_col in basis_to_pivot_col.items():
            if p_col != -1 and (orig_row >> p_col) & 1:
                basis_indices_needed.append(b_idx)
                
        if not basis_indices_needed:
            outputs.append(n)
        else:
            # Get the wires for these basis vectors
            wires = [basis_wires[b_idx] for b_idx in basis_indices_needed]
            out_wire = xor_list(wires)
            outputs.append(out_wire)
            
    return {'gates': gates, 'outputs': outputs}
