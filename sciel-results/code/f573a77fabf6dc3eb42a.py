import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0
    
    if not matrix:
        return {'gates': [], 'outputs': []}
        
    m = len(matrix)
    n = len(matrix[0])
    
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
        
    # Gaussian Elimination with Weight Minimization
    sorted_rows = sorted(rows, key=lambda x: bin(x).count('1'))
    basis = []
    
    for r in sorted_rows:
        if r == 0:
            continue
        changed = True
        while changed:
            changed = False
            for b in basis:
                new_r = r ^ b
                if bin(new_r).count('1') < bin(r).count('1'):
                    r = new_r
                    changed = True
                    break
        if r != 0:
            basis.append(r)
            
    # Local Search
    improved = True
    k = len(basis)
    while improved and time.time() < deadline:
        improved = False
        for i in range(k):
            for j in range(i+1, k):
                new_i = basis[i] ^ basis[j]
                if bin(new_i).count('1') < bin(basis[i]).count('1'):
                    basis[i] = new_i
                    improved = True

    # Precompute XOR values
    xor_vals = [0] * (1 << k)
    for c in range(1, 1 << k):
        lsb = c & -c
        bit_idx = lsb.bit_length() - 1
        prev_c = c ^ lsb
        xor_vals[c] = xor_vals[prev_c] ^ basis[bit_idx]
        
    val_to_coeff = {}
    for c in range(1 << k):
        val_to_coeff[xor_vals[c]] = c
        
    coeff_list = [val_to_coeff.get(r, 0) for r in rows]
    
    next_wire = n + 1
    gates = []
    
    def build_xor_tree(wires, next_wire, gates):
        if not wires:
            return n, 0
        if len(wires) == 1:
            return wires[0], 0
            
        mid = len(wires) // 2
        left_wires = wires[:mid]
        right_wires = wires[mid:]
        
        left_out, left_gates = build_xor_tree(left_wires, next_wire, gates)
        next_wire += left_gates
        
        right_out, right_gates = build_xor_tree(right_wires, next_wire, gates)
        next_wire += right_gates
        
        gates.append([left_out, right_out])
        out_wire = next_wire
        next_wire += 1
        
        return out_wire, left_gates + right_gates + 1

    basis_wires = []
    for b in basis:
        inputs = [i for i in range(n) if (b >> i) & 1]
        out_wire, num_gates = build_xor_tree(inputs, next_wire, gates)
        basis_wires.append(out_wire)
        next_wire += num_gates
        
    outputs = []
    for c in coeff_list:
        sel_wires = [basis_wires[j] for j in range(k) if (c >> j) & 1]
        out_wire, num_gates = build_xor_tree(sel_wires, next_wire, gates)
        outputs.append(out_wire)
        next_wire += num_gates
        
    return {'gates': gates, 'outputs': outputs}
