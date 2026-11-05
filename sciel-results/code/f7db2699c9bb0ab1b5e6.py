"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit (not strictly enforced in this heuristic,
                     but we aim for shallow circuits).
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
                     Wire indices: 0..n-1 are inputs, n is constant zero, n+1+k is the k-th gate output.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s limit
    
    # Convert matrix to numpy array of integers
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs  # constant zero wire
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Represent each row as a frozenset of input indices where the bit is 1.
    row_sets = []
    for i in range(n_rows):
        s = frozenset(np.where(A[i] == 1)[0])
        row_sets.append(s)
    
    # Baseline gate count: sum(max(0, popcount(row) - 1)) for rows with popcount > 0
    def baseline_gate_count():
        total = 0
        for s in row_sets:
            k = len(s)
            if k > 1:
                total += k - 1
        return total
    
    base_count = baseline_gate_count()
    
    # Greedy sub-expression sharing
    # State:
    #   row_reprs[i] = frozenset of wire indices for row i
    #   wire_sets[w] = frozenset of original input indices that wire w represents
    #   gates = list of [a, b] pairs
    #   next_wire = next available wire index
    
    row_reprs = [frozenset(s) for s in row_sets]
    wire_sets = {i: frozenset([i]) for i in range(n_inputs)}
    wire_sets[zero_wire] = frozenset()
    gates = []
    next_wire = zero_wire + 1
    
    # Track which wires are "live" (used in at least one row representation)
    # and their usage count
    
    def get_usage_counts():
        """Count how many times each wire is used across all row representations."""
        usage = {}
        for rs in row_reprs:
            for w in rs:
                usage[w] = usage.get(w, 0) + 1
        return usage
    
    def total_remaining_gates():
        """Compute total gates needed if we XOR each row's wires independently."""
        total = 0
        for rs in row_reprs:
            k = len(rs)
            if k > 1:
                total += k - 1
        return total
    
    # Greedy loop
    max_iterations = 10000
    iter_count = 0
    improved = True
    
    while improved and iter_count < max_iterations:
        if time.time() > deadline:
            break
        iter_count += 1
        improved = False
        
        # Find all pairs of wires that appear together in at least one row
        pair_count = {}
        for rs in row_reprs:
            elems = list(rs)
            for i in range(len(elems)):
                for j in range(i+1, len(elems)):
                    a, b = elems[i], elems[j]
                    if a > b:
                        a, b = b, a
                    pair = (a, b)
                    pair_count[pair] = pair_count.get(pair, 0) + 1
        
        if not pair_count:
            break
        
        # For each candidate pair, compute the benefit
        best_pair = None
        best_benefit = 0
        
        for (a, b), count in pair_count.items():
            if count < 2:
                continue  # Only consider pairs that appear in at least 2 rows
            
            # Compute what the new wire would represent
            set_a = wire_sets[a]
            set_b = wire_sets[b]
            new_set = set_a ^ set_b  # Symmetric difference = XOR
            
            # Check if this new wire is already computed
            # (i.e., is there an existing wire with the same set?)
            # For simplicity, we'll allow duplicates but this could be optimized
            
            # Benefit: if we create wire w = a XOR b, then in each row containing both a and b,
            # we can replace {a, b} with {w}, reducing the row size by 1.
            # The gate count for a row of size k is k-1 (if k>0), so reducing k by 1 saves 1 gate.
            # But we add 1 gate for the new wire.
            # Net benefit = count - 1 (if count >= 2)
            
            # However, we need to be careful: if a row has size 1, it needs 0 gates.
            # If a row has size 2, it needs 1 gate. Replacing {a,b} with {w} keeps size 1, still 0 gates.
            # So the saving is exactly 1 per row where both a and b are present and the row size > 1.
            
            # Actually, more precisely:
            # Before: row has size k, needs max(0, k-1) gates
            # After: row has size k-1, needs max(0, k-2) gates
            # Saving = max(0, k-1) - max(0, k-2) = 1 if k >= 2, else 0
            
            # So benefit = (number of rows containing both a and b with size >= 2) - 1 (for the new gate)
            
            rows_with_pair = 0
            for rs in row_reprs:
                if a in rs and b in rs and len(rs) >= 2:
                    rows_with_pair += 1
            
            benefit = rows_with_pair - 1
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = (a, b, new_set)
        
        if best_pair is None:
            break
        
        a, b, new_set = best_pair
        
        # Create the new wire
        new_wire = next_wire
        next_wire += 1
        wire_sets[new_wire] = new_set
        gates.append([a, b])
        
        # Update row representations
        for i in range(n_rows):
            rs = row_reprs[i]
            if a in rs and b in rs:
                new_rs = rs - {a, b} | {new_wire}
                row_reprs[i] = frozenset(new_rs)
        
        improved = True
    
    # Build the final output
    outputs = []
    for i in range(n_rows):
        rs = row_reprs[i]
        if len(rs) == 0:
            outputs.append(zero_wire)
        elif len(rs) == 1:
            outputs.append(list(rs)[0])
        else:
            # Need to XOR the wires in rs
            # But we already counted these in the gate structure
            # Actually, we need to add gates for the final XORs
            # This is a problem: our greedy approach only added intermediate gates,
            # but we still need to combine the final wires for each row.
            
            # Let me reconsider: the total gate count should be:
            # (gates already added) + sum(max(0, len(row_reprs[i]) - 1) for i)
            
            # So we need to add the final combination gates.
            elems = list(rs)
            current = elems[0]
            for e in elems[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
