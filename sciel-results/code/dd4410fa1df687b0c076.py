"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Check for zero rows
    row_popcounts = A.sum(axis=1)
    zero_rows = set(np.where(row_popcounts == 0)[0])
    
    # Represent each row as a set of wire indices
    row_wires = []
    for i in range(n_rows):
        if i in zero_rows:
            row_wires.append(set())
        else:
            row_wires.append(set(np.where(A[i] == 1)[0].tolist()))
    
    # Map from wire to its underlying frozenset of input indices
    wire_set = {}
    for i in range(n_inputs):
        wire_set[i] = frozenset([i])
    wire_set[zero_wire] = frozenset()
    
    # Depth tracking: depth of each wire
    wire_depth = {}
    for i in range(n_inputs):
        wire_depth[i] = 0
    wire_depth[zero_wire] = 0
    
    def elapsed_ok():
        return time.time() < deadline
    
    def try_greedy(seed=None):
        """Run greedy synthesis with a given random seed for tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        gates = []
        next_wire = zero_wire + 1
        
        # Local copies
        rw = [set(s) for s in row_wires]
        ws = dict(wire_set)
        wd = dict(wire_depth)
        
        max_iter = 20000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs in rows with size >= 2
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = rw[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        # Skip if either is zero wire
                        if a == zero_wire or b == zero_wire:
                            continue
                        # Skip if they represent the same set (XOR would be 0)
                        if ws[a] == ws[b]:
                            continue
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair with tie-breaking
            best_pair = None
            best_benefit = 0
            best_depth = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                # New depth would be max(depth[a], depth[b]) + 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            
            # Compute the set for the new wire
            new_set = ws[a] ^ ws[b]
            if len(new_set) == 0:
                # This would produce zero, skip
                continue
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            gates.append([a, b])
            
            # Update all rows containing both a and b
            for i in range(n_rows):
                rs = rw[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Build outputs
        outputs = []
        for i in range(n_rows):
            rs = rw[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                # Sort by depth to minimize overall depth
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    # Check depth
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        # Can't add more gates due to depth, just output as is
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    ws[current] = ws.get(current, frozenset())  # placeholder
                    wd[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    def try_basis_greedy(seed=None):
        """
        Use Gaussian elimination to find a basis, then greedily optimize
        the computation of basis rows with shared subexpressions.
        """
        if seed is not None:
            random.seed(seed)
        
        # Step 1: Find a basis of independent rows via Gaussian elimination
        A_copy = A.copy().astype(np.int8)
        n_r, n_c = A_copy.shape
        
        pivot_rows = []  # indices of rows that are part of the basis
        pivot_cols = []  # pivot column for each basis row
        basis_vectors = []  # the actual basis vectors (as sets)
        
        # Perform elimination to identify pivot rows
        col = 0
        for row in range(n_r):
            if col >= n_c:
                break
            # Find a row with 1 in column col from row onwards
            pivot = None
            for r in range(row, n_r):
                if A_copy[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue
            
            # Swap rows
            if pivot != row:
                A_copy[[row, pivot]] = A_copy[[pivot, row]]
            
            # Eliminate this column from all other rows
            for r in range(n_r):
                if r != row and A_copy[r, col] == 1:
                    A_copy[r] = (A_copy[r] + A_copy[row]) % 2
            
            pivot_rows.append(row)
            pivot_cols.append(col)
            basis_vectors.append(set(np.where(A_copy[row] == 1)[0].tolist()))
            col += 1
        
        rank = len(pivot_rows)
        
        if rank == 0:
            return try_greedy(seed)
        
        # Step 2: Express each row as a combination of basis vectors
        # After elimination, A_copy has the basis in pivot_rows
        # Each original row i can be expressed as XOR of basis vectors
        # where the coefficient is A_copy[i, pivot_cols[j]]
        
        # But we need the original rows, not the eliminated ones
        # Let's redo: track which original rows form the basis
        
        # Actually, let's use a different approach:
        # Find independent rows and express all rows in terms of them
        
        # Use numpy to find rank and basis
        A_int = A.copy()
        
        # Find a maximal set of linearly independent rows
        independent_indices = []
        independent_vectors = []  # as bit arrays
        dependent_coeffs = {}  # row_index -> list of basis indices
        
        # Process rows in order, maintaining a row-echelon form
        echelon = []  # list of (pivot_col, bit_array)
        
        for i in range(n_rows):
            if i in zero_rows:
                dependent_coeffs[i] = []
                continue
            
            v = A_int[i].copy()
            # Reduce v by existing echelon vectors
            coeffs = []
            for (pc, ev) in echelon:
                if v[pc] == 1:
                    v = (v + ev) % 2
                    coeffs.append(pc)
            
            if v.sum() == 0:
                # Dependent
                dependent_coeffs[i] = coeffs
            else:
                # Find pivot
                pc = int(np.argmax(v))
                echelon.append((pc, v.copy()))
                independent_indices.append(i)
                independent_vectors.append(v.copy())
                dependent_coeffs[i] = []
        
        if len(independent_indices) == 0:
            return try_greedy(seed)
        
        # Step 3: Build circuit
        # Compute each independent (basis) row using shared subexpressions
        # Then express each dependent row as XOR of basis rows
        
        gates = []
        next_wire = zero_wire + 1
        
        # Local state for greedy optimization of basis computation
        ws = dict(wire_set)
        wd = dict(wire_depth)
        
        # Represent each basis row as a set of wire indices (initially inputs)
        basis_reprs = []
        for i in independent_indices:
            if i in zero_rows:
                basis_reprs.append(set())
            else:
                basis_reprs.append(set(np.where(A[i] == 1)[0].tolist()))
        
        # Greedily optimize the computation of basis rows
        max_iter = 10000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs in basis rows with size >= 2
            pair_count = defaultdict(int)
            for i in range(len(basis_reprs)):
                rs = basis_reprs[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        if a == zero_wire or b == zero_wire:
                            continue
                        if ws[a] == ws[b]:
                            continue
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair
            best_pair = None
            best_benefit = 0
            best_depth = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            
            new_set = ws[a] ^ ws[b]
            if len(new_set) == 0:
                continue
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            gates.append([a, b])
            
            # Update all basis rows containing both a and b
            for i in range(len(basis_reprs)):
                rs = basis_reprs[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Step 4: Compute final wires for each basis row
        basis_wires = []
        for i in range(len(basis_reprs)):
            rs = basis_reprs[i]
            if len(rs) == 0:
                basis_wires.append(zero_wire)
            elif len(rs) == 1:
                basis_wires.append(list(rs)[0])
            else:
                elems = list(rs)
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wd[current] = new_depth
                basis_wires.append(current)
        
        # Step 5: Express each output row as XOR of basis rows
        outputs = []
        for i in range(n_rows):
            if i in zero_rows:
                outputs.append(zero_wire)
                continue
            
            if i in dependent_coeffs and len(dependent_coeffs[i]) > 0:
                # Dependent row: XOR of basis rows
                coeff_cols = dependent_coeffs[i]
                # Map pivot columns to basis indices
                # We need to find which basis rows correspond to these pivot columns
                # Build mapping from pivot column to basis index
                pivot_to_basis = {}
                for j, (pc, ev) in enumerate(echelon):
                    pivot_to_basis[pc] = j
                
                basis_indices = []
                for pc in coeff_cols:
                    if pc in pivot_to_basis:
                        basis_indices.append(pivot_to_basis[pc])
                
                if not basis_indices:
                    outputs.append(zero_wire)
                elif len(basis_indices) == 1:
                    outputs.append(basis_wires[basis_indices[0]])
                else:
                    # XOR all the basis wires
                    current = basis_wires[basis_indices[0]]
                    for bi in basis_indices[1:]:
                        gates.append([current, basis_wires[bi]])
                        current = next_wire
                        next_wire += 1
                    outputs.append(current)
            else:
                # Independent row: use its basis wire
                # Find its index in independent_indices
                idx = independent_indices.index(i)
                outputs.append(basis_wires[idx])
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try multiple strategies and keep the best
    best_result = None
    best_gate_count = float('inf')
    
    # First try the pure greedy approach with several seeds
    for seed in [0, 1, 42, 123, 7, 99, 256, 777]:
        if not elapsed_ok():
            break
        result = try_greedy(seed=seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    # Try the basis-based approach
    for seed in [0, 1, 42]:
        if not elapsed_ok():
            break
        result = try_basis_greedy(seed=seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    if best_result is None:
        # Fallback: independent row synthesis
        gates = []
        next_wire = zero_wire + 1
        outputs = []
        for i in range(n_rows):
            rs = row_wires[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                current = elems[0]
                for e in elems[1:]:
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        return {'gates': gates, 'outputs': outputs}
    
    return best_result

# EVOLVE-BLOCK-END
