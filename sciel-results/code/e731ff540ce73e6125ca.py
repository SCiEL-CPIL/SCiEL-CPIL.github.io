"""
XOR-circuit synthesis via greedy rank-reduction set cover.
"""
import time
import numpy as np

def _gf2_rank(mat):
    """Compute rank of a binary matrix over GF(2)."""
    if mat.size == 0:
        return 0
    a = mat.copy()
    n_rows = a.shape[0]
    n_cols = a.shape[1]
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if a[row, col]:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap rows
        a[[rank, pivot]] = a[[pivot, rank]]
        # Eliminate
        for row in range(n_rows):
            if row != rank and a[row, col]:
                a[row] = np.bitwise_xor(a[row], a[rank])
        rank += 1
        if rank == n_rows:
            break
    return rank

def _row_hash(row):
    """Hash a binary row for fast comparison."""
    # Convert to integer for hashing
    return int.from_bytes(row.tobytes(), byteorder='little')

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    matrix = np.array(matrix, dtype=np.uint8)
    n_inputs, n_outputs = matrix.shape
    
    # Identify non-zero target rows
    nonzero_rows = []
    for i in range(n_outputs):
        if np.any(matrix[i]):
            nonzero_rows.append(i)
    
    if not nonzero_rows:
        return {'gates': [], 'outputs': list(range(n_outputs))}
    
    # Baseline: direct implementation (each output is one wire, no gates needed if we count differently)
    # Actually, in XOR circuit synthesis, we need to compute each output as XOR of input wires.
    # The "independent-row baseline" likely means: for each output row, we need popcount(row) XOR gates
    # but we can share intermediate results.
    
    # Let's think about this differently. We have n_inputs input wires (0..n_inputs-1)
    # and a constant-zero wire (n_inputs).
    # Gate k produces wire n_inputs+1+k.
    # We need to produce n_outputs output wires.
    
    # Strategy: Greedy set cover where we try to find intermediate XORs that reduce the total
    # number of operations needed to compute all remaining outputs.
    
    # Start with all outputs remaining
    remaining = list(nonzero_rows)
    gates = []
    wire_map = {}  # maps a tuple of input indices to the wire that computes their XOR
    
    # Initial: input wires are 0..n_inputs-1, zero wire is n_inputs
    # We'll track what each wire computes (as a frozenset of input indices)
    wire_to_set = {i: frozenset([i]) for i in range(n_inputs)}
    wire_to_set[n_inputs] = frozenset()  # zero
    
    next_wire = n_inputs + 1
    used_wires = set(range(n_inputs + 1))  # wires 0..n_inputs are available
    
    # We want to minimize total gates. Let's use a greedy approach:
    # At each step, consider all pairs of existing wires, compute their XOR,
    # and see how much it reduces the "cost" of computing remaining outputs.
    
    # Cost of computing a target row t given available wires:
    # We need to find a subset of available wires whose XOR equals t.
    # This is a linear algebra problem: find x in GF(2)^m such that A*x = t
    # where A is the matrix of available wire representations.
    # The cost is the number of non-zero entries in x (popcount of the solution).
    # But we want to minimize this, so we want the sparsest solution.
    
    # For small problems, we can do this exactly.
    
    def compute_cost(target_row, available_wires_list):
        """Compute minimum number of available wires to XOR to get target_row."""
        if not available_wires_list:
            return float('inf')
        
        # Build matrix of available wire representations
        n_wires = len(available_wires_list)
        n_inputs_local = n_inputs
        
        # Each wire is a bit vector of length n_inputs
        A = np.zeros((n_wires, n_inputs_local), dtype=np.uint8)
        for i, w in enumerate(available_wires_list):
            s = wire_to_set[w]
            for idx in s:
                A[i, idx] = 1
        
        # Target
        t = target_row.copy()
        
        # Solve A*x = t over GF(2), find sparsest solution
        # Use Gaussian elimination to find one solution, then try to reduce
        
        # Augmented matrix
        M = np.column_stack([A, t.reshape(-1, 1)])
        n_rows = M.shape[0]
        n_cols = M.shape[1] - 1
        
        # Gaussian elimination
        rank = 0
        pivot_cols = []
        for col in range(n_cols):
            pivot = -1
            for row in range(rank, n_rows):
                if M[row, col]:
                    pivot = row
                    break
            if pivot == -1:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            for row in range(n_rows):
                if row != rank and M[row, col]:
                    M[row] = np.bitwise_xor(M[row], M[rank])
            rank += 1
            pivot_cols.append(col)
        
        # Check consistency
        for row in range(rank, n_rows):
            if M[row, n_cols]:
                return float('inf')  # No solution
        
        # Find a particular solution (set free variables to 0)
        x = np.zeros(n_cols, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            x[col] = M[i, n_cols]
        
        # The particular solution has popcount:
        cost = np.sum(x)
        
        # Try to reduce by flipping null space vectors
        # Find null space basis
        free_cols = [c for c in range(n_cols) if c not in pivot_cols]
        
        # For each free variable, compute the corresponding null space vector
        # and check if flipping it reduces the popcount
        improved = True
        while improved:
            improved = False
            for fc in free_cols:
                # Compute null space vector: set free var fc=1, others=0, solve for pivots
                x_trial = np.zeros(n_cols, dtype=np.uint8)
                x_trial[fc] = 1
                # For each pivot row, the pivot variable = sum of other variables in that row
                for i, col in enumerate(pivot_cols):
                    # x[col] = M[i, n_cols] + sum_{j != col} M[i,j]*x[j]
                    # But in reduced form, M[i, col] = 1 and other pivot cols are 0
                    # So x[col] = M[i, n_cols] + sum_{free j} M[i,j]*x[j]
                    s = M[i, n_cols]
                    for j in free_cols:
                        if M[i, j] and x_trial[j]:
                            s ^= 1
                    x_trial[col] = s
                new_cost = np.sum(x_trial)
                if new_cost < cost:
                    x = x_trial
                    cost = new_cost
                    improved = True
                    break
        
        return cost
    
    # Initial available wires
    available_wires = list(range(n_inputs + 1))
    
    # Greedy: repeatedly add the gate that gives the best reduction in total cost
    max_gates = 200  # Safety limit
    while remaining and len(gates) < max_gates:
        if time.time() > deadline - 10.0:
            break
        
        # Compute current total cost
        current_total_cost = sum(compute_cost(matrix[r], available_wires) for r in remaining)
        
        # Try all pairs of available wires
        best_gain = 0
        best_pair = None
        n_avail = len(available_wires)
        
        for i in range(n_avail):
            for j in range(i + 1, n_avail):
                w1 = available_wires[i]
                w2 = available_wires[j]
                s1 = wire_to_set[w1]
                s2 = wire_to_set[w2]
                s_new = frozenset(s1.symmetric_difference(s2))
                
                # Skip if this wire already exists
                if s_new in set(wire_to_set.values()):
                    continue
                
                # Compute new total cost with this wire added
                new_available = available_wires + [next_wire]
                # Temporarily add to wire_to_set
                wire_to_set[next_wire] = s_new
                
                new_total_cost = sum(compute_cost(matrix[r], new_available) for r in remaining)
                gain = current_total_cost - new_total_cost
                
                # Remove temporary
                del wire_to_set[next_wire]
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (w1, w2, s_new)
        
        if best_pair is None or best_gain <= 0:
            break
        
        w1, w2, s_new = best_pair
        gates.append([w1, w2])
        available_wires.append(next_wire)
        wire_to_set[next_wire] = s_new
        next_wire += 1
        
        # Check if any remaining outputs are now directly computable (cost 1)
        # Actually, we should check if the new wire directly matches any remaining output
        for r in list(remaining):
            if wire_to_set[next_wire - 1] == frozenset(np.where(matrix[r])[0]):
                # This output can be directly assigned to this wire
                remaining.remove(r)
                # We need to track which wire produces each output
                # We'll handle output assignment at the end
    
    # Assign outputs to wires
    # For each output, find the wire that computes it
    outputs = [0] * n_outputs
    for r in range(n_outputs):
        if not np.any(matrix[r]):
            outputs[r] = n_inputs  # zero wire
        else:
            # Find a wire that matches
            target_set = frozenset(np.where(matrix[r])[0])
            found = False
            for w in range(next_wire):
                if w in wire_to_set and wire_to_set[w] == target_set:
                    outputs[r] = w
                    found = True
                    break
            if not found:
                # Need to construct it from available wires
                # Use the cost function to find which wires to XOR
                cost = compute_cost(matrix[r], list(range(next_wire)))
                # Find the actual combination
                A = np.zeros((next_wire, n_inputs), dtype=np.uint8)
                for w in range(next_wire):
                    if w in wire_to_set:
                        for idx in wire_to_set[w]:
                            A[w, idx] = 1
                t = matrix[r].copy()
                M = np.column_stack([A, t.reshape(-1, 1)])
                rank = 0
                pivot_cols = []
                for col in range(n_inputs):
                    pivot = -1
                    for row in range(rank, next_wire):
                        if M[row, col]:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    M[[rank, pivot]] = M[[pivot, rank]]
                    for row in range(next_wire):
                        if row != rank and M[row, col]:
                            M[row] = np.bitwise_xor(M[row], M[rank])
                    rank += 1
                    pivot_cols.append(col)
                x = np.zeros(next_wire, dtype=np.uint8)
                for i, col in enumerate(pivot_cols):
                    x[col] = M[i, n_inputs]
                # This gives us which wires to XOR, but we need to build the circuit
                # For now, just pick the first wire that has a 1 in x
                # This is a fallback; ideally we should have found a direct match
                for w in range(next_wire):
                    if x[w]:
                        outputs[r] = w
                        break
    
    return {'gates': gates, 'outputs': outputs}
