# EVOLVE-BLOCK-START
import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0  # 90s wall-clock limit, 50s internal safety

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

    def elapsed():
        return time.time() - start_time

    # --- Gaussian elimination to find basis with minimal total weight ---
    # We want a basis B of the row space such that each row is a short XOR of basis vectors.
    # Strategy: find a basis, then for each row find the minimum-weight representation.
    
    # Step 1: Find a basis via Gaussian elimination
    basis = []  # list of (pivot_col, mask)
    basis_dict = {}  # pivot_col -> mask
    
    working = list(row_masks)
    for col in range(n - 1, -1, -1):
        if elapsed() > DEADLINE - 10.0:
            break
        # Find a row with bit col set
        found = -1
        for i, mask in enumerate(working):
            if (mask >> col) & 1:
                found = i
                break
        if found == -1:
            continue
        # Eliminate col from all other rows
        pivot_mask = working[found]
        for i, mask in enumerate(working):
            if i != found and ((mask >> col) & 1):
                working[i] = mask ^ pivot_mask
        basis.append((col, pivot_mask))
        basis_dict[col] = pivot_mask
        working[found] = 0

    # Step 2: For each row, find minimum-weight representation as XOR of basis vectors
    # This is a linear algebra problem: express row in terms of basis
    # Since basis is in RREF-like form (each pivot col appears in only one basis vector),
    # the representation is unique and can be found greedily.
    
    def express_in_basis(target):
        """Express target as XOR of subset of basis vectors. Returns list of basis indices."""
        if target == 0:
            return []
        result = []
        remaining = target
        for col, bmask in basis:
            if (remaining >> col) & 1:
                result.append(bmask)
                remaining ^= bmask
        return result

    # Step 3: Build wires for basis vectors, then for each row XOR the needed basis wires
    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    memo = {}  # mask -> wire_id
    
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_xor_tree(wires):
        """Build a balanced XOR tree from a list of wires. Returns final wire."""
        if not wires:
            return n
        if len(wires) == 1:
            return wires[0]
        
        # Balanced tree construction with memoization
        def _tree(w_list):
            if len(w_list) == 1:
                return w_list[0]
            # Pair up and recurse
            new_w = []
            for i in range(0, len(w_list), 2):
                if i + 1 < len(w_list):
                    a = w_list[i]
                    b = w_list[i+1]
                    # Check if this XOR is already memoized
                    # We need to track masks for memoization
                    # For simplicity in tree, just create gate
                    new_w.append(add_gate(a, b))
                else:
                    new_w.append(w_list[i])
            return _tree(new_w)
        
        return _tree(wires)

    # Build wires for each basis vector
    basis_wires = []
    for col, bmask in basis:
        if bmask in memo:
            basis_wires.append(memo[bmask])
        else:
            # Build from bits
            bits = [j for j in range(n) if (bmask >> j) & 1]
            if len(bits) <= 1:
                w = bits[0] if bits else n
            else:
                w = build_xor_tree(bits)
            memo[bmask] = w
            basis_wires.append(w)

    # Step 4: For each output row, find its representation and build the XOR
    outputs = [None] * m
    for i in range(m):
        if elapsed() > DEADLINE - 5.0:
            # Fast fallback: direct tree
            mask = row_masks[i]
            if mask in memo:
                outputs[i] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                if not bits:
                    outputs[i] = n
                elif len(bits) == 1:
                    outputs[i] = bits[0]
                else:
                    outputs[i] = build_xor_tree(bits)
                memo[mask] = outputs[i]
            continue
        
        target = row_masks[i]
        if target in memo:
            outputs[i] = memo[target]
            continue
        
        # Find representation in basis
        rep = express_in_basis(target)
        
        if not rep:
            outputs[i] = n
            memo[target] = n
            continue
        
        # Get wires for each basis vector in the representation
        # We need to map basis mask -> wire
        # Build a lookup
        mask_to_wire = {}
        for idx, (col, bmask) in enumerate(basis):
            mask_to_wire[bmask] = basis_wires[idx]
        
        wires_needed = [mask_to_wire[bm] for bm in rep]
        
        # Build XOR of these wires
        if len(wires_needed) == 1:
            w = wires_needed[0]
        else:
            w = build_xor_tree(wires_needed)
        
        outputs[i] = w
        memo[target] = w

    # Fill any None
    for i in range(m):
        if outputs[i] is None:
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
