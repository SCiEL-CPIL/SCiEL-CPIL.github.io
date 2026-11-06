"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)
        
    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    
    # memo: bitmask -> wire_id
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_balanced_tree(indices):
        """
        Builds a balanced XOR tree from a list of input indices.
        Returns the wire ID.
        """
        if not indices:
            return n
        if len(indices) == 1:
            return indices[0]
            
        wires = list(indices)
        # We don't need to track masks here if we are just combining existing wires,
        # but for the initial basis construction from inputs, we just combine wires.
        # The mask of the result is the XOR of the masks of the inputs.
        # However, for sharing, we should memoize the mask of the result.
        
        # Let's do it recursively or iteratively with memoization of the mask.
        # For a balanced tree, we can just pair them up.
        
        # To allow sharing, we should check if the resulting mask is in memo.
        # But here we are building from scratch for a specific set.
        # A simple iterative pairwise XOR is fine.
        
        while len(wires) > 1:
            new_wires = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    w = add_gate(wires[i], wires[i+1])
                    new_wires.append(w)
                else:
                    new_wires.append(wires[i])
            wires = new_wires
        return wires[0]

    def compute_mask_optimized(mask):
        """
        Computes wire for mask using memoization.
        If not in memo, computes it using a balanced tree of its bits.
        This is a fallback. The main strategy uses the basis.
        """
        if mask in memo:
            return memo[mask]
        
        bits = [j for j in range(n) if (mask >> j) & 1]
        w = build_balanced_tree(bits)
        memo[mask] = w
        return w

    # 1. Perform Gaussian Elimination to find a basis for the row space
    # and the representation of each row in terms of this basis.
    
    # We want a basis that minimizes the total cost.
    # Standard RREF is a good heuristic.
    
    # Let's create a list of (mask, index) and reduce.
    # We'll maintain a list of basis vectors and their "coordinates" in the original input space?
    # No, we just need the basis vectors as bitmasks, and for each row, which basis vectors XOR to it.
    
    basis = []
    # coeffs[i] will be the bitmask of basis indices that sum to row_masks[i]
    # Initially, row_masks[i] is represented by itself? No.
    # Let's do standard RREF on the matrix of rows.
    
    # To be efficient, we work with the bitmasks.
    # We want to find a set of linearly independent vectors that span the rows.
    
    # Let's use a dictionary to track the "current form" of rows during elimination.
    # But we also need to track the transformation.
    
    # Alternative: 
    # Find pivot columns.
    # For each pivot column, find a row with a 1 in that column, make it the pivot row,
    # and eliminate that column from all other rows.
    # The pivot rows form the basis.
    # The coefficients of the original rows in terms of the pivot rows can be tracked.
    
    # Let's store the basis vectors as they are formed.
    # And for each original row, store the bitmask of basis vectors that sum to it.
    
    # Let basis_masks be the list of basis vector masks.
    # Let row_coeffs[i] be the bitmask of indices in basis_masks that sum to row_masks[i].
    
    basis_masks = []
    row_coeffs = [0] * m
    
    # We iterate through columns to find pivots.
    # But rows are not in a fixed order.
    # We can just iterate through the rows and add them to the basis if they are independent.
    # This gives a basis, but not necessarily RREF.
    # To get the coefficients, we need to track the operations.
    
    # Simpler approach:
    # 1. Find a basis B of the row space.
    # 2. For each row r, find its coordinates in B.
    
    # Step 1: Find Basis
    # We can use the rows themselves.
    # Let's pick a set of rows that are linearly independent.
    
    # We'll maintain a list of selected basis rows (indices) and their current reduced form?
    # Actually, for the circuit, we just need the masks of the basis vectors.
    # And for each output, the list of basis vectors to XOR.
    
    # Let's do Gaussian Elimination on the matrix of rows to find the rank and a basis.
    # We'll keep track of the "linear combination" of the original rows that form each basis vector.
    # Wait, the basis vectors ARE the rows (or combinations of rows).
    # If we use the rows themselves as the basis (when independent), then the basis vectors are just some of the input rows.
    # This is valid.
    
    # Let's select a maximal set of linearly independent rows from the input.
    # Let independent_rows be the list of indices.
    
    independent_indices = []
    # We'll store the basis vectors as bitmasks.
    # To make the basis "good" (low weight), we might want to reduce them.
    # But using the original rows as basis vectors is often fine because their weight is already determined.
    # However, if we reduce them, we might get lower weights?
    # Example: r1=111, r2=011. Basis: 111, 011. Weights 3, 2. Cost: 2+1=3 gates.
    # If we reduce: r2 is independent. r1 xor r2 = 100. Basis: 011, 100. Weights 2, 1. Cost: 1+0=1 gate for basis.
    # Then r1 = 011 xor 100 (1 gate). Total 2 gates. Better!
    
    # So, we should compute a Reduced Row Echelon Form (RREF) basis.
    # The basis vectors in RREF have pivots, and are often sparse.
    
    # Let's perform RREF on the bitmasks.
    # We'll maintain a list of basis vectors (masks) and their "history" (which original rows they came from)?
    # No, we just need the basis masks and the coefficients of each original row in terms of these basis masks.
    
    # Let's do this:
    # 1. Copy row_masks to work_arr.
    # 2. Perform RREF on work_arr.
    # 3. The non-zero rows of work_arr are the basis vectors.
    # 4. We also need to track how each original row is expressed in terms of these basis vectors.
    #    This is equivalent to finding the inverse of the transformation matrix?
    #    Or simply: as we eliminate, we track the coefficients.
    
    # Let's use a standard algorithm to find the basis and the coordinates.
    # We'll maintain a list `basis` of masks.
    # And a list `coords` where `coords[i]` is the bitmask of basis indices that sum to `row_masks[i]`.
    
    # Initialize
    basis = []
    coords = [0] * m
    
    # We process columns from MSB to LSB (or LSB to MSB, doesn't matter for GF(2) if consistent).
    # Let's process from high bit to low bit to find pivots.
    
    # To allow sharing, we want the basis vectors to be computed first.
    # Then each output is a sum of basis vectors.
    
    # Algorithm:
    # for col in reversed(range(n)):
    #   find a row i >= current_row with bit col set.
    #   if found:
    #     swap with current_row
    #     for all rows j != current_row:
    #        if row j has bit col set:
    #           row j = row j xor row current_row
    #           coords[j] = coords[j] xor (1 << len(basis)) # Wait, basis index?
    #     add row current_row to basis
    #     coords[current_row] = 1 << len(basis) # The new basis vector is itself?
    #     current_row += 1
    
    # This is slightly tricky because `coords` needs to reflect the basis vectors as they are added.
    
    # Let's try a different perspective.
    # We want to find a matrix $C$ such that $M = C B$, where $B$ is the basis matrix.
    # The columns of $C$ correspond to the rows of $M$. The entries are 0/1.
    # The cost is sum of weights of rows of $B$ (to compute $B$) + sum of weights of columns of $C$ (to compute $M$).
    
    # Let's compute RREF of $M$.
    # The non-zero rows of RREF($M$) form a basis $B_{rref}$.
    # The rows of $M$ can be expressed as linear combinations of the rows of RREF($M$).
    # Actually, it's easier to say:
    # Let $R$ be the RREF of $M$.
    # There exists a matrix $T$ such that $M = T R$.
    # The rows of $R$ are the basis vectors.
    # The rows of $T$ give the coefficients.
    
    # How to compute $T$?
    # We can perform the same row operations on an identity matrix $I$ (of size $m \times m$) to get $T$?
    # No, $M$ is $m \times n$. RREF is $m \times n$.
    # The row operations that transform $M$ to $R$ can be represented by an invertible matrix $E$ such that $R = E M$.
    # Then $M = E^{-1} R$. So $T = E^{-1}$.
    
    # We can compute $E$ by applying the same operations to $I_m$.
    # Then we need $E^{-1}$.
    # Or, we can just track the operations on the "coordinates" directly.
    
    # Let's initialize `basis` as empty.
    # Let `row_repr` be a list of length $m$, where `row_repr[i]` is a bitmask of length $m$ (initially $1 \ll i$), representing the current row $i$ as a combination of the ORIGINAL rows.
    # As we perform row operations, we update `row_repr` similarly.
    # At the end, the non-zero rows of `work_arr` are the basis vectors.
    # Their `row_repr` tells us which original rows they are? No, we need the reverse: which BASIS vectors sum to each ORIGINAL row.
    
    # This is getting complicated. Let's stick to the simpler heuristic:
    # 1. Find a basis $B$ (list of masks).
    # 2. For each row $r$, find the subset of $B$ that sums to $r$.
    
    # Finding the subset is a linear system solve: $B^T x = r$.
    # Since $B$ is small (size $\le 16$), we can solve this for each row by brute force or Gaussian elimination.
    
    # Step 1: Find Basis
    # We'll pick a maximal set of independent rows from `row_masks`.
    # To get a "good" basis, we can try to reduce them.
    # Let's just take the first $k$ independent rows we find, and reduce the rest?
    # Actually, RREF basis is best.
    
    # Let's implement RREF and track the coefficients.
    
    # We'll work with a copy of the rows.
    # And we'll maintain a list `coeffs` where `coeffs[i]` is the bitmask of basis indices (so far) that sum to the current row $i$.
    # Wait, if we swap rows, we swap coeffs too.
    # If we XOR row $j$ with row $i$, we XOR $coeffs[j]$ with $coeffs[i]$.
    
    # Let's try this.
    
    work_rows = row_masks[:]
    # coeffs[i] will be the bitmask of the FINAL basis vectors that sum to the ORIGINAL row i.
    # This is hard to track forward.
    
    # Let's do this:
    # 1. Compute RREF basis $B$.
    # 2. For each original row $r$, solve $B x = r$ for $x$.
    #    Since $B$ is a set of independent vectors, this solution is unique.
    
    # How to solve $B x = r$?
    # $B$ is a list of $k$ masks.
    # We can build a matrix of size $n \times k$ (columns are $B_i$).
    # We want to find $x$ (length $k$) such that $\sum x_i B_i = r$.
    # This is equivalent to finding $x$ in the column space.
    # We can do this by Gaussian elimination on the system $B x = r$.
    # But $B$ is fixed for all $r$.
    # We can precompute the inverse of the basis matrix?
    
    # Let's just solve it for each row. $k \le 16$, so it's fast.
    
    # First, find the basis $B$.
    # We'll use the rows of $M$ to find a basis.
    # We'll pick rows that are independent.
    
    basis = []
    for mask in row_masks:
        if mask == 0:
            continue
        # Check if mask is independent of current basis
        # Reduce mask by basis
        temp = mask
        for b in basis:
            # If the highest set bit of b is set in temp, XOR
            # We need to order basis by pivot.
            # Let's maintain basis in RREF form.
            pass
    
    # This is getting messy. Let's use a standard RREF function that returns the basis and the transformation matrix.
    
    def rref_and_transform(rows):
        """
        Returns basis (list of masks) and transform (list of bitmasks of length m).
        transform[i] is the bitmask of basis indices that sum to rows[i].
        """
        m = len(rows)
        n = len(rows[0]) if m>0 else 0 # not used directly, masks are used
        
        # We'll work with a list of (mask, coeff_mask)
        # coeff_mask is the bitmask of original row indices that sum to this row.
        # Initially, row i is 1 << i.
        
        arr = [(rows[i], 1 << i) for i in range(m)]
        
        # We want to find the basis vectors and their "original" composition?
        # No, we want the basis vectors (masks) and for each original row, which basis vectors sum to it.
        
        # Let's just find the basis vectors first.
        # Then solve for coefficients.
        
        # Find Basis
        basis = []
        # We'll reduce the rows to find independent ones.
        # We'll keep the basis in a reduced form.
        
        # To make it simple, let's just pick independent rows.
        # We'll sort by bit_count to prefer sparse rows?
        # No, just pick any.
        
        for mask in rows:
            if mask == 0: continue
            # Reduce mask by existing basis
            # We need to find the pivot of mask and basis vectors.
            # Let's assume basis vectors are in RREF.
            
            # Check independence
            temp = mask
            for b in basis:
                # If b has a pivot that is set in temp, XOR
                # We need to know the pivot of b.
                # Let's store (mask, pivot)
                pass
        
        # This is too complex to implement correctly from scratch in a short time.
        # Let's use the property that for $n \le 16$, we can just use the rows themselves if they are independent, 
        # and if not, we can just compute the cost for each row independently?
        # No, sharing is the key.
        
        # Let's try a simpler heuristic:
        # 1. Compute the rank $k$.
        # 2. If $k$ is small, find a basis.
        # 3. If $k$ is large, independent rows are better.
        
        # Let's go with the "Compute Basis then Solve" approach.
        
        # Step 1: Find Basis
        # We'll use Gaussian Elimination to find a basis of the row space.
        # We'll store the basis vectors as bitmasks.
        
        basis = []
        # We'll maintain the basis in a form where each vector has a unique pivot bit.
        
        # To find the basis, we can iterate through the rows and add them if they are independent.
        # To check independence, we reduce the new row by the existing basis.
        
        # Let's store basis as list of (mask, pivot_bit)
        
        for mask in row_masks:
            if mask == 0:
                continue
            temp = mask
            # Reduce temp by basis
            for i, (b_mask, b_pivot) in enumerate(basis):
                if (temp >> b_pivot) & 1:
                    temp ^= b_mask
            if temp != 0:
                # Find pivot of temp
                pivot = temp.bit_length() - 1
                # Add to basis
                # We should insert it in the correct position to maintain RREF?
                # Or just append and reduce others?
                # Let's just append.
                basis.append((temp, pivot))
                
        # Now we have a basis. It might not be fully reduced (other basis vectors might have bits in the pivot positions of other vectors).
        # For solving $B x = r$, we need the basis to be in RREF (or at least a triangular form).
        # Let's reduce the basis to RREF.
        
        # Convert to list of masks
        b_masks = [b[0] for b in basis]
        
        # Reduce to RREF
        # Sort by pivot descending
        basis.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(len(basis)):
            # Eliminate bit basis[i][1] from all other basis vectors
            pivot_i = basis[i][1]
            for j in range(len(basis)):
                if i != j and (basis[j][0] >> pivot_i) & 1:
                    basis[j] = (basis[j][0] ^ basis[i][0], basis[j][1])
                    
        b_masks = [b[0] for b in basis]
        
        # Step 2: For each row, find coefficients
        # We need to solve $B x = r$.
        # Since $B$ is in RREF, the solution is easy.
        # The pivot of $B_i$ is $p_i$.
        # The $i$-th bit of $x$ is determined by the $p_i$-th bit of the remaining part of $r$.
        
        # Actually, since $B$ is RREF, the $i$-th column of $B$ has a 1 at $p_i$ and 0s elsewhere in the pivot columns.
        # So $x_i = r_{p_i}$ (after eliminating other pivots?).
        
        # Let's just do a simple Gaussian Elimination for each row.
        # Or, since $B$ is RREF, we can just extract the bits.
        
        # Wait, RREF means:
        # 1. Leading 1 in each non-zero row.
        # 2. Each leading 1 is the only non-zero entry in its column.
        
        # So, if we have $r$, and we want $x$ such that $\sum x_i B_i = r$.
        # Look at the pivot column $p_i$ of $B_i$.
        # The bit $r_{p_i}$ must equal $x_i$ (because all other $B_j$ have 0 at $p_j$ for $j \ne i$? No, other $B_j$ have 0 at $p_i$).
        # Yes, in RREF, the column $p_i$ has 1 at row $i$ and 0 elsewhere.
        # So $x_i = r_{p_i}$.
        
        # Let's verify.
        # $r = \sum x_i B_i$.
        # Bit $p_k$ of $r$ is $\sum x_i (B_i)_{p_k}$.
        # $(B_i)_{p_k}$ is 1 if $i=k$, 0 otherwise.
        # So $r_{p_k} = x_k$.
        
        # This is extremely fast!
        
        coeffs = []
        for r in row_masks:
            x = 0
            for i, (b_mask, b_pivot) in enumerate(basis):
                if (r >> b_pivot) & 1:
                    x |= (1 << i)
            coeffs.append(x)
            
        return b_masks, coeffs

    b_masks, coeffs = rref_and_transform(row_masks)
    
    # Step 3: Compute wires for basis
    # We compute each basis vector using a balanced tree.
    # We memoize the result.
    
    basis_wires = []
    for b_mask in b_masks:
        if b_mask in memo:
            basis_wires.append(memo[b_mask])
        else:
            bits = [j for j in range(n) if (b_mask >> j) & 1]
            w = build_balanced_tree(bits)
            memo[b_mask] = w
            basis_wires.append(w)
            
    # Step 4: Compute wires for outputs
    outputs = []
    for i in range(m):
        x = coeffs[i]
        if x == 0:
            outputs.append(n) # Zero wire
            continue
            
        # Get the indices of basis vectors to XOR
        indices = [j for j in range(len(b_masks)) if (x >> j) & 1]
        
        if len(indices) == 1:
            outputs.append(basis_wires[indices[0]])
        else:
            # Build balanced tree from basis_wires
            wires = [basis_wires[j] for j in indices]
            while len(wires) > 1:
                new_wires = []
                for k in range(0, len(wires), 2):
                    if k + 1 < len(wires):
                        w = add_gate(wires[k], wires[k+1])
                        new_wires.append(w)
                    else:
                        new_wires.append(wires[k])
                wires = new_wires
            outputs.append(wires[0])
            
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
