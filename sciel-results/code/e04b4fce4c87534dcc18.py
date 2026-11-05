# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using Gaussian Elimination to find a minimal basis
    and then constructing a shared sub-circuit for the basis vectors.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    mat = np.array(matrix, dtype=np.uint8)
    n_outputs, n_inputs = mat.shape

    if n_outputs == 0 or n_inputs == 0:
        return {'gates': [], 'outputs': [n_inputs] * n_outputs}

    # Convert rows to bitmasks (Python integers to avoid overflow)
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m

    input_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]

    # Step 1: Gaussian Elimination to find a basis for the row space.
    # We maintain a list of basis vectors. Each basis vector is a bitmask.
    # We also maintain a mapping from pivot column to basis vector index for quick lookup.
    basis = []
    pivot_to_idx = {}
    
    # To allow efficient reduction of new vectors against the basis,
    # we keep the basis in a state where each vector has a unique pivot,
    # and all other basis vectors have 0 at that pivot position.
    # This is essentially Reduced Row Echelon Form (RREF).
    
    for v in input_masks:
        v_current = v
        # Reduce v_current using existing basis vectors
        # We iterate through existing basis vectors.
        # If v_current has a 1 in the pivot column of a basis vector, XOR it out.
        # To do this efficiently, we can iterate through the basis list.
        # Since n_inputs is small (<=16), this is fast.
        
        # Optimization: Keep basis sorted by pivot? Or just iterate.
        # Let's just iterate through the current basis list.
        for b in basis:
            # Find pivot of b
            # We can store pivots to avoid re-calculating, but n is small.
            # Let's find pivot of b.
            pb = b.bit_length() - 1 if b else -1
            if pb >= 0 and (v_current >> pb) & 1:
                v_current ^= b
        
        if v_current != 0:
            # Find pivot of v_current
            pv = v_current.bit_length() - 1
            # Eliminate this pivot from all existing basis vectors
            new_basis = []
            for b in basis:
                pb = b.bit_length() - 1 if b else -1
                if pb >= 0 and (b >> pv) & 1:
                    b ^= v_current
                new_basis.append(b)
            basis = new_basis
            basis.append(v_current)
            pivot_to_idx[pv] = len(basis) - 1

    # Now `basis` is a linearly independent set spanning the row space.
    # We need to map each basis vector to a wire in the circuit.
    
    # Wire mapping:
    # 0..n_inputs-1: input wires
    # n_inputs: zero wire
    # n_inputs+1...: generated wires
    
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    gates = []
    next_wire = n_inputs + 1
    
    def create_wire(mask):
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
        if mask == 0:
            return n_inputs
            
        # Find the bits set in mask
        bits = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                bits.append(idx)
            m >>= 1
            idx += 1
            
        if not bits:
            return n_inputs
            
        if len(bits) == 1:
            w = wire_map[1 << bits[0]]
            wire_map[mask] = w
            return w
            
        # Split into two roughly equal parts to minimize depth
        mid = len(bits) // 2
        m1 = 0
        m2 = 0
        for b in bits[:mid]:
            m1 |= (1 << b)
        for b in bits[mid:]:
            m2 |= (1 << b)
            
        w1 = create_wire(m1)
        w2 = create_wire(m2)
        
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        wire_map[mask] = w
        return w

    # Build wires for all basis vectors
    basis_wires = []
    for b in basis:
        w = create_wire(b)
        basis_wires.append(w)

    # Step 2: Express each output row as a linear combination of the basis vectors.
    # Since the basis is in RREF (each pivot column has 1 only in its corresponding basis vector),
    # the coefficient for basis vector k (with pivot p_k) is simply the bit at p_k in the output row.
    
    # We need to map pivot column to the index in the basis list.
    # We have pivot_to_idx from the elimination step.
    
    outputs = []
    for i in range(n_outputs):
        y = input_masks[i]
        if y == 0:
            outputs.append(n_inputs)
            continue
        
        # Find which basis vectors are needed
        # The coefficient for basis vector at index idx (with pivot p) is (y >> p) & 1
        # We need to XOR the wires of all basis vectors where the coefficient is 1.
        
        w = None
        # Iterate through basis vectors
        # To be efficient, we can iterate through the set bits of y, check if they are pivots,
        # and if so, include the corresponding basis wire.
        
        # However, y might have 1s in non-pivot columns.
        # The condition for including basis vector k is:
        # The bit of y at the pivot position of basis vector k is 1.
        
        # Let's collect the wires to XOR
        wires_to_xor = []
        
        # Iterate through all basis vectors to check their pivot bits in y
        # n_basis is at most n_inputs (16), so this is fast.
        for idx, b in enumerate(basis):
            pivot = b.bit_length() - 1
            if (y >> pivot) & 1:
                wires_to_xor.append(basis_wires[idx])
                
        if not wires_to_xor:
            outputs.append(n_inputs)
        else:
            if len(wires_to_xor) == 1:
                outputs.append(wires_to_xor[0])
            else:
                # XOR the wires
                w = wires_to_xor[0]
                for j in range(1, len(wires_to_xor)):
                    gates.append([w, wires_to_xor[j]])
                    w = next_wire
                    next_wire += 1
                outputs.append(w)

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
