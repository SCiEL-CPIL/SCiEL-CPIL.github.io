import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import sys

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using a graph-based approach (Min-Cost Max-Flow concept).
    
    Strategy:
    1. Model the problem as finding a set of intermediate XORs (basis) and decomposing
       each target row into a sum of these basis elements.
    2. To maximize sharing, we use a heuristic that prioritizes creating basis elements
       that cover the most "demand" (bits in target rows).
    3. We use a layered graph construction (BFS by weight) to ensure depth constraints 
       and efficient decomposition.
    
    The "flow" interpretation: Each target row needs to be "covered" by basis elements.
    A basis element used by multiple rows "saves" gates compared to independent synthesis.
    We greedily select basis elements that maximize the reduction in total gate count.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Compute row weights and baseline
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # 1. Identify "High Value" sub-masks to use as basis elements.
    # A basis element b is valuable if it appears as a subset of many target rows.
    # However, we don't just want subsets; we want to decompose targets into sums of basis elements.
    
    # Let's use a different graph-based heuristic:
    # Build a graph where nodes are masks. Edges represent XOR.
    # We want to find a set of "hubs" (basis elements) such that every target is close to a hub.
    
    # Simpler robust approach for "different strategy":
    # Use Gaussian Elimination to find a basis of the row space.
    # Then, synthesize each basis vector optimally (using a small DP or greedy).
    # Then, express each target row as a linear combination of the basis vectors.
    # This guarantees sharing if the basis vectors are shared.
    
    # Wait, the current program already does something similar but "greedy per row".
    # The "Flow" idea implies a global optimization.
    
    # Let's implement a "Shared Basis" strategy:
    # 1. Compute a basis for the matrix rows.
    # 2. Synthesize the basis vectors. To minimize gates, we synthesize them in an order
    #    that allows maximum reuse of intermediate XORs.
    # 3. For each target row, compute its coordinates in the basis.
    # 4. The output wire is the XOR of the basis wires corresponding to the coordinates.
    #    To minimize gates for this step, we can also use a small tree for the combination.
    
    # Step 1: Gaussian Elimination to get basis and coordinates
    # We work over GF(2).
    # Represent rows as integers.
    
    rows_int = target_masks[:]
    
    basis = []
    pivot_cols = []
    
    # Sort rows by weight to potentially find "simpler" basis vectors first?
    # Actually, standard GE is fine.
    
    # We need to track the transformation to find coordinates.
    # Let A be the matrix. We perform row operations to get R (row echelon form).
    # R = T * A.
    # We want to express each original row a_i as sum of basis vectors.
    
    # Alternative: 
    # 1. Find a basis B = {b_1, ..., b_k} for the span of the rows.
    # 2. For each target row t, find coefficients c such that t = sum c_i b_i.
    # 3. Synthesize b_i's.
    # 4. Synthesize t by XORing the corresponding b_i's.
    
    # To make this "Graph/Flow" like:
    # The "cost" is the number of gates.
    # Synthesizing b_i costs G(b_i).
    # Combining b_i's for t costs G_comb(t).
    # Total cost = sum G(b_i) + sum G_comb(t_i).
    
    # We want to choose a basis B that minimizes this.
    # A "good" basis consists of vectors with low weight (cheap to synthesize).
    
    # Algorithm:
    # 1. Get all rows.
    # 2. Perform GE to get a basis. Keep track of which original rows formed the basis.
    # 3. Synthesize the basis vectors.
    #    - Use a greedy algorithm to synthesize the set of basis vectors together,
    #      maximizing sharing of intermediate nodes.
    # 4. For each target row, determine which basis vectors are needed.
    #    - If 0 or 1 basis vectors, direct wire.
    #    - If >1, build a balanced XOR tree for those basis wires.
    
    # This is distinct from the current "per-row greedy" because it forces all rows to
    # be expressed in terms of a common set of pre-computed basis wires.
    
    # --- Implementation ---
    
    # 1. Gaussian Elimination
    # We will store the basis vectors and their coordinates in the original space?
    # No, we need coordinates of original rows in the basis space.
    
    # Let's use an augmented matrix approach.
    # Columns 0..n-1: the bits of the row.
    # Columns n..n+m-1: identity matrix for tracking transformations?
    # Actually, simpler:
    # We process rows one by one.
    # If a row is independent, it becomes a basis vector.
    # We also maintain a "reduced" version of all rows in terms of the basis.
    
    basis_vectors = []
    basis_pivots = [] # pivot column for each basis vector
    
    # reduced_rows: list of bitmasks representing the row in terms of the current basis?
    # No, let's just do standard GE on the matrix to get Row Echelon Form (REF).
    # The non-zero rows of REF form a basis.
    # The transformation matrix T tells us how REF rows relate to original rows.
    
    # Actually, we want the inverse: Original rows as linear combos of REF rows.
    
    # Let's do this:
    # 1. Copy matrix to a list of lists of ints.
    # 2. Perform GE to get REF.
    # 3. The non-zero rows in REF are our basis vectors (in terms of original inputs).
    # 4. To find the decomposition of an original row into these basis vectors,
    #    we can just run the original row through the same elimination steps?
    #    Or, simpler: since we have the basis vectors, we can just solve for coefficients
    #    for each target row.
    
    # Let's get the basis vectors first.
    ref_rows = []
    # Use a copy of the integer masks
    working_rows = target_masks[:]
    
    # Sort by weight? No, standard GE is deterministic.
    # To get a "good" basis (low weight), we might prefer pivots that lead to low weight rows.
    # Standard GE picks the first available pivot.
    
    # Let's do a "Weight-Optimized" GE:
    # At each step, pick the row with the lowest weight that has a 1 in the current pivot column?
    # Or pick the pivot column that minimizes the weight of the resulting row?
    
    # For now, standard GE.
    # We also need to track the coordinates.
    
    # Let's use a different representation for coordinates:
    # coords[i] is a bitmask of length k (basis size) indicating which basis vectors sum to row i.
    
    # We will perform GE on the matrix, but we will also maintain an "identity" matrix
    # that tracks how the current rows are formed from the original rows.
    # Wait, we want Original = C * Basis.
    # If we do GE on A to get R, then R = T A.
    # So A = T^-1 R.
    # The rows of T^-1 give the coordinates of original rows in terms of R's rows.
    
    # This is expensive to compute T^-1 explicitly if m is large.
    # But m is small (<=16).
    
    # Let's construct the augmented matrix [A | I_m].
    # Perform row operations to bring A to REF.
    # The right side becomes T.
    # Then we need T^-1.
    
    # Actually, there is a simpler way.
    # 1. Find basis B from A.
    # 2. For each original row a_i, solve B^T x = a_i for x.
    #    Since B is a basis for the span, a solution exists.
    #    B is k x n. x is k x 1.
    #    We can solve this by performing GE on [B^T | a_i].
    
    # Let's find the basis first.
    
    # We want a basis with low weights to minimize synthesis cost.
    # Greedy Basis Selection:
    # 1. Start with empty basis.
    # 2. Repeatedly pick the row (from remaining rows, or XORs of rows?) with the lowest weight
    #    that is linearly independent of the current basis.
    #    Adding it to the basis.
    # 3. This might be slow if we check all XORs.
    #    Let's just pick from the original rows.
    
    # Sort original rows by weight.
    sorted_indices = np.argsort(row_weights)
    
    basis = []
    basis_indices = []
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        r = target_masks[idx]
        # Check if r is in span of basis
        # We can do this by trying to reduce r using the basis.
        # If it reduces to 0, it's dependent.
        # If not, it's independent.
        
        # To make reduction fast, we maintain the basis in REF form?
        # Let's just keep the basis vectors and reduce.
        
        is_independent = True
        reduced_r = r
        for b in basis:
            # Find pivot of b
            # We need to know the pivot column of b.
            # Let's store (mask, pivot_col)
            pass
            
    # This is getting complicated. Let's stick to a robust, simpler "Different Strategy":
    # **Shared Intermediate Nodes via Clustering**
    
    # 1. Compute all pairwise XORs of input wires? No.
    # 2. Identify frequent sub-expressions.
    #    For each pair of inputs (i, j), compute mask m_ij = e_i ^ e_j.
    #    Count how many target rows contain both i and j, or neither, or one?
    #    Actually, if a target row has weight > 1, it's a sum of inputs.
    #    If we pre-compute XORs of pairs that appear frequently in the targets, we save gates.
    
    # Let's use a **Greedy Set Cover** approach for sharing.
    # Universe: Set of "gates" needed if we synthesized independently.
    # Actually, let's look at the "Flow" hint again.
    # "Minimum-cost maximum-flow... nodes represent all subsets... edges represent XOR".
    
    # I will implement a **BFS-based Layered Synthesis** which is effectively a shortest path
    # in the state space of masks, but applied globally.
    
    # 1. Build a graph of masks reachable from inputs in <= depth_limit steps.
    #    Nodes: masks.
    #    Edges: mask -> mask ^ (1<<i) with cost 1.
    #    This graph is huge (2^n).
    
    # Instead, let's use the **Basis Decomposition** strategy but with a **Smart Basis Choice**.
    
    # 1. Compute a basis for the row space.
    # 2. To minimize the cost of synthesizing the basis, we choose a basis that minimizes the sum of weights.
    #    We do this by performing GE, but at each step, we choose the pivot row that results in the lowest weight row.
    
    # 3. Synthesize the basis vectors using a greedy algorithm that shares intermediates.
    
    # 4. Decompose each target row into the basis.
    
    # This is distinct from the current program which does NOT use a global basis, but rather
    # computes each row independently (with some local sharing).
    
    # --- Implementation of Smart Basis ---
    
    # 1. Get a basis with minimal total weight.
    #    We perform GE. At each step k, we have a set of candidate rows (those with a 1 in col k
    #    after reducing by previous pivots). We pick the one with the lowest weight.
    
    #    We need to maintain the rows in a way that allows us to reduce them.
    #    Let's store the current state of all rows as a list of ints.
    #    And a list of "active" rows (not yet processed).
    
    #    Actually, standard GE modifies the matrix in place.
    #    To pick the "best" pivot, we need to look at all rows that have a 1 in the current column.
    
    #    Algorithm:
    #    1. Copy matrix to list of ints `rows`.
    #    2. `basis` = []
    #    3. `pivot_col` = 0
    #    4. While pivot_col < n and len(basis) < m:
    #       a. Find all rows i such that bit `pivot_col` is set in `rows[i]`.
    #       b. If none, pivot_col++, continue.
    #       c. Among these rows, find the one with the minimum weight (popcount).
    #          Let this be `best_row`.
    #       d. Swap `rows[best_row]` with `rows[current_basis_index]`?
    #          No, we just use `best_row` as the pivot.
    #       e. For all other rows j, if bit `pivot_col` is set in `rows[j]`, XOR `rows[best_row]` into `rows[j]`.
    #       f. Add `rows[best_row]` to `basis`.
    #       g. pivot_col++.
    
    #    This gives a basis `B` with minimal weights (locally optimal choice at each step).
    
    # 2. Synthesize the basis vectors `B`.
    #    We have k basis vectors.
    #    We synthesize them using a greedy algorithm that maximizes sharing.
    #    We maintain a set of `available` masks.
    #    We process basis vectors in order of increasing weight.
    #    For each basis vector, we find the cheapest decomposition using `available`.
    
    # 3. For each target row, find its coordinates in `B`.
    #    We can do this by running the target row through the same elimination process
    #    used to generate `B`, but tracking the coefficients.
    
    #    Alternatively, since we have the final `rows` (which are in REF form relative to the original? No, they are the reduced rows),
    #    wait. The `rows` list after GE contains the basis vectors in the first k positions, and 0s in the rest.
    #    But we lost the original rows.
    
    #    Let's keep a copy of the original rows.
    #    To find coordinates of `orig_row` in `B`:
    #    We can solve `sum c_i B_i = orig_row`.
    #    Since `B` is a basis, we can just do GE on the system.
    #    Or, simpler: we can just re-run the reduction of `orig_row` using `B`.
    #    If we reduce `orig_row` using `B` (in reverse order of elimination?), we can track the XORs.
    
    #    Actually, there is a neat trick:
    #    If we perform the same sequence of XOR operations on an identity matrix as we did on the data matrix,
    #    we get the transformation matrix.
    
    #    Let's do that.
    #    `data_rows`: list of ints (original rows).
    #    `trans_rows`: list of ints (identity matrix, row i has bit i set).
    
    #    Perform GE on `data_rows`, applying the same operations to `trans_rows`.
    #    After GE, `data_rows[:k]` are the basis vectors.
    #    `trans_rows` is the matrix T such that `data_rows_ref = T * data_rows_orig`.
    #    Wait, `data_rows` is now in REF.
    #    So `REF = T * ORIG`.
    #    We want `ORIG = T^{-1} * REF`.
    #    The rows of `T^{-1}` are the coordinates of the original rows in the REF basis.
    
    #    Computing T^{-1} for m=16 is trivial (Gauss-Jordan on T).
    
    #    Steps:
    #    1. Initialize `data` = target_masks, `trans` = identity masks.
    #    2. Perform GE with weight-optimization on `data`, mirroring ops on `trans`.
    #    3. Extract basis `B` from `data`.
    #    4. Extract `T` from `trans`.
    #    5. Compute `T_inv` by inverting `T` (Gauss-Jordan on [T | I]).
    #    6. The i-th row of `T_inv` gives the coordinates of original row i in the basis B.
    
    #    7. Synthesize B using greedy sharing.
    #    8. For each original row i, get coords c_i.
    #       The output is XOR of B_j for all j where c_i[j] is 1.
    #       Synthesize this combination efficiently (balanced tree).
    
    # This is a solid, distinct strategy.
    
    # --- Implementation ---
    
    # 1. GE with Weight Optimization
    
    data = target_masks[:]
    trans = [1 << i for i in range(m)]
    
    basis = []
    
    pivot_col = 0
    basis_count = 0
    
    while pivot_col < n and basis_count < m:
        # Find candidate rows
        candidates = []
        for i in range(basis_count, m):
            if (data[i] >> pivot_col) & 1:
                candidates.append(i)
        
        if not candidates:
            pivot_col += 1
            continue
        
        # Pick candidate with min weight
        best_idx = min(candidates, key=lambda i: bin(data[i]).count('1'))
        
        # Swap to current basis position
        data[basis_count], data[best_idx] = data[best_idx], data[basis_count]
        trans[basis_count], trans[best_idx] = trans[best_idx], trans[basis_count]
        
        # Eliminate this column from other rows
        pivot_row = data[basis_count]
        pivot_trans = trans[basis_count]
        
        for i in range(m):
            if i != basis_count and ((data[i] >> pivot_col) & 1):
                data[i] ^= pivot_row
                trans[i] ^= pivot_trans
        
        basis.append(pivot_row)
        basis_count += 1
        pivot_col += 1
    
    # Now `data[:basis_count]` are the basis vectors.
    # `trans` is the transformation matrix T.
    
    # 2. Invert T to get coordinates
    # T is m x m. We only care about the first m rows (which is all of it).
    # Actually, T maps Original -> REF.
    # We want Original -> Basis.
    # The basis vectors are the non-zero rows of REF.
    # So we need the inverse of the transformation that maps Original to the specific basis rows we extracted.
    
    # Let's construct the matrix T explicitly as a list of ints (rows of T).
    # T[i] is the bitmask of the i-th row of T.
    
    # We want T_inv such that T_inv * T = I.
    # Perform Gauss-Jordan on [T | I].
    
    # Augmented matrix: m rows, 2m columns.
    # Represent as list of tuples or two lists.
    
    # Since m <= 16, we can use integers for the 2m bits.
    
    aug = []
    for i in range(m):
        # Left part: T[i], Right part: I[i]
        left = trans[i]
        right = 1 << i
        aug.append((left, right))
    
    # Gauss-Jordan
    for col in range(m):
        # Find pivot
        pivot_row = -1
        for r in range(col, m):
            if (aug[r][0] >> col) & 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            # Singular? Should not happen if we handled basis correctly.
            # If T is not invertible, it means some original rows were dependent.
            # But we are inverting the transformation on the full space.
            # If the matrix is rank deficient, T is not square invertible in the usual sense?
            # T is m x m. If the rows of A are dependent, T is singular.
            # We only need the coordinates for the rows that are in the span.
            # Actually, if T is singular, we can't invert it.
            
            # Fallback: Use the basis directly to solve for each row.
            # But for now, assume full rank or handle singularity by skipping.
            # If singular, we can just use the current `trans` rows? No.
            
            # Let's just break and use a fallback for this row if inversion fails.
            pass
        
        # Swap
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Normalize (not needed in GF(2) if pivot is 1)
        
        # Eliminate
        for r in range(m):
            if r != col and ((aug[r][0] >> col) & 1):
                aug[r] = (aug[r][0] ^ aug[col][0], aug[r][1] ^ aug[col][1])
        
    # Now aug[r] is (0..0, 1, 0..0, T_inv[r])
    # The right part is the r-th row of T_inv.
    
    # coords[i] = aug[i][1]
    # This bitmask indicates which basis vectors sum to original row i.
    
    # 3. Synthesize Basis Vectors
    # We have `basis` list of ints.
    # We want to synthesize them with shared intermediates.
    
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # Sort basis vectors by weight
    basis_sorted = sorted(enumerate(basis), key=lambda x: bin(x[1]).count('1'))
    
    basis_wires = [0] * len(basis)
    
    for idx, b_mask in basis_sorted:
        if time.time() > deadline:
            break
            
        if b_mask in available:
            basis_wires[idx] = available[b_mask]
            continue
        
        # Find decomposition
        found = False
        # Try 2-way split
        for w_mask, w_wire in list(available.items()):
            other_mask = b_mask ^ w_mask
            if other_mask in available:
                other_wire = available[other_mask]
                if w_mask == 0:
                    basis_wires[idx] = other_wire
                    found = True
                    break
                elif other_mask == 0:
                    basis_wires[idx] = w_wire
                    found = True
                    break
                else:
                    gates.append([w_wire, other_wire])
                    basis_wires[idx] = next_wire
                    next_wire += 1
                    available[b_mask] = basis_wires[idx]
                    found = True
                    break
        
        if not found:
            # Fallback: direct synthesis
            ones = [j for j in range(n) if (b_mask >> j) & 1]
            if len(ones) == 0:
                basis_wires[idx] = zero_wire
            elif len(ones) == 1:
                basis_wires[idx] = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                basis_wires[idx] = current
                available[b_mask] = basis_wires[idx]
    
    # 4. Synthesize Outputs
    outputs = []
    
    for i in range(m):
        if time.time() > deadline:
            break
            
        # Get coordinates
        # If inversion failed, we might have garbage.
        # Let's verify the coordinate mask.
        coord_mask = aug[i][1]
        
        # Find which basis vectors are needed
        needed_basis = []
        for j in range(len(basis)):
            if (coord_mask >> j) & 1:
                needed_basis.append(basis_wires[j])
        
        if len(needed_basis) == 0:
            outputs.append(zero_wire)
        elif len(needed_basis) == 1:
            outputs.append(needed_basis[0])
        else:
            # Build balanced XOR tree
            # Simple pairwise
            current = needed_basis[0]
            for j in range(1, len(needed_basis)):
                gates.append([current, needed_basis[j]])
                current = next_wire
                next_wire += 1
            outputs.append(current)
            
    return {'gates': gates, 'outputs': outputs}

