import numpy as np
import time

def _popcount(x):
    return int(bin(x).count('1'))

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    matrix = np.array(matrix, dtype=np.uint8) % 2
    m, n = matrix.shape
    zero_wire = n
    # basis vectors stored as Python ints (bit i = column i)
    rows = [int(''.join(map(str, matrix[i])), 2) if False else None for i in range(m)]
    # build int per row: column j is bit j
    row_ints = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i, j] & 1:
                v |= (1 << j)
        row_ints.append(v)
    
    # Gaussian elimination to get RREF and basis + combination
    # We'll work with rows as ints; track pivot columns.
    # Standard: process columns left to right (bit 0 = col 0).
    basis = []  # list of (pivot_col, basis_int, combo_int) where combo_int has bits over original rows? 
    # Better: track basis as list of ints (the vector), and for each original row track which basis it uses.
    # Let's do elimination producing row-echelon, keep basis vectors and the coefficient of each original row in terms of basis.
    
    # Approach: maintain a list `vecs` of current row vectors (ints), initially row_ints.
    # Also maintain for each current row a "history" = which original rows it's a combo of? Not needed.
    # We need: basis vectors (final independent ones) and for each ORIGINAL row, the combination over basis.
    # Easiest: do elimination on an augmented system. Treat each original row as e_i in an extra m-bit space.
    
    # Represent each working row as (vector_int, marker_int) where marker has bits over original rows (m bits).
    work = [(row_ints[i], 1 << i) for i in range(m)]
    basis_vecs = []   # vector ints
    basis_markers = []  # marker ints (which original rows combine to form this basis vector)
    pivot_cols = set()
    
    # Sort columns: we'll iterate col from 0..n-1
    for col in range(n):
        bit = 1 << col
        # find a row with this bit set among remaining (not yet used as pivot)
        # We'll keep work as list; find index where vector has bit and marker not zero, and vector not in basis.
        idx = None
        for k in range(len(work)):
            v, mk = work[k]
            if mk == 0:
                continue
            if v & bit:
                idx = k
                break
        if idx is None:
            continue
        v, mk = work[idx]
        # eliminate this bit from all other work rows
        for k in range(len(work)):
            if k == idx:
                continue
            vk, mkk = work[k]
            if vk & bit:
                work[k] = (vk ^ v, mkk ^ mk)
        # Now v is a basis vector with pivot col
        basis_vecs.append(v)
        basis_markers.append(mk)
        pivot_cols.add(col)
    
    r = len(basis_vecs)
    
    # Now each basis_vecs[i] is independent. Each original row j is a combination of basis vectors.
    # We need combo[j] in GF(2)^r such that row_ints[j] = XOR of basis_vecs[i] where combo[j][i]=1.
    # Since basis markers tell us basis_vecs[i] = XOR of original rows in marker[i], we can solve.
    # Actually easier: we already eliminated; the original rows are expressible. Let's just solve linear system.
    # Build matrix B (r x n) = basis_vecs. Solve for each original row.
    # Use the same elimination structure: we have basis in row-echelon with pivots. 
    # For each original row, reduce it by basis vectors to get combo.
    
    combo = []
    for j in range(m):
        v = row_ints[j]
        c = 0
        # reduce using basis in reverse order of pivot (highest pivot first) to match echelon
        # basis_vecs are in increasing pivot col order. To reduce, go from last to first.
        for i in range(r-1, -1, -1):
            pv = basis_vecs[i]
            # pivot col of basis i = lowest set bit? In our construction pivot is the col we found, which is the lowest set bit of v at that time.
            # Actually the pivot col is the col we processed; v's lowest set bit equals that col.
            pcol = (pv & -pv).bit_length() - 1
            if v & (1 << pcol):
                v ^= pv
                c |= (1 << i)
        # v should be 0
        combo.append(c)
    
    # Greedily reduce basis vector weights: for each basis i, try XOR with other basis to reduce popcount
    # Do a few passes.
    def reduce_basis():
        changed = True
        while changed:
            changed = False
            for i in range(r):
                for j in range(r):
                    if i == j:
                        continue
                    cand = basis_vecs[i] ^ basis_vecs[j]
                    if cand == 0:
                        continue
                    if _popcount(cand) < _popcount(basis_vecs[i]):
                        basis_vecs[i] = cand
                        basis_markers[i] ^= basis_markers[j]
                        changed = True
                        break
    reduce_basis()
    
    # Recompute combo after basis change? The basis changed, so combos must be recomputed.
    # Recompute combo by solving again with new basis (still independent, same pivot structure? pivots may change).
    # Safer: recompute combo by linear solve using new basis.
    def solve_combo(row_int):
        v = row_int
        c = 0
        # need pivot cols of new basis
        pcols = []
        for i in range(r):
            pv = basis_vecs[i]
            pcol = (pv & -pv).bit_length() - 1
            pcols.append(pcol)
        # reduce from highest pivot
        order = sorted(range(r), key=lambda i: pcols[i], reverse=True)
        for i in order:
            pv = basis_vecs[i]
            pcol = pcols[i]
            if v & (1 << pcol):
                v ^= pv
                c |= (1 << i)
        return c
    combo = [solve_combo(row_ints[j]) for j in range(m)]
    
    # Build circuit
    # Wires: 0..n-1 inputs, n zero, gates produce n+1, n+2, ...
    gates = []
    wire_count = n + 1  # next wire index
    def xor_gate(a, b):
        nonlocal wire_count
        w = wire_count
        wire_count += 1
        gates.append([a, b])
        return w
    def compute_vector(vec_int):
        # XOR all set bits
        bits = []
        for j in range(n):
            if vec_int & (1 << j):
                bits.append(j)
        if not bits:
            return zero_wire
        if len(bits) == 1:
            return bits[0]
        w = bits[0]
        for b in bits[1:]:
            w = xor_gate(w, b)
        return w
    
    basis_wires = []
    for i in range(r):
        basis_wires.append(compute_vector(basis_vecs[i]))
    
    outputs = []
    for j in range(m):
        c = combo[j]
        # XOR basis wires where c has bit
        idxs = [i for i in range(r) if c & (1 << i)]
        if not idxs:
            outputs.append(zero_wire)
        elif len(idxs) == 1:
            outputs.append(basis_wires[idxs[0]])
        else:
            w = basis_wires[idxs[0]]
            for i in idxs[1:]:
                w = xor_gate(w, basis_wires[i])
            outputs.append(w)
    
    # Verify
    # Simulate
    wire_val = [0]*(wire_count)
    for j in range(n):
        wire_val[j] = 1 << j  # input j
    # zero wire stays 0
    for k in range(len(gates)):
        a, b = gates[k]
        wire_val[n+1+k] = wire_val[a] ^ wire_val[b]
    ok = True
    for j in range(m):
        if wire_val[outputs[j]] != row_ints[j]:
            ok = False
            break
    if not ok:
        # fallback: independent rows
        gates = []
        wire_count = n + 1
        def xor_gate2(a, b):
            nonlocal wire_count
            w = wire_count
            wire_count += 1
            gates.append([a, b])
            return w
        outputs = []
        for j in range(m):
            bits = [k for k in range(n) if row_ints[j] & (1 << k)]
            if not bits:
                outputs.append(zero_wire)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = xor_gate2(w, b)
                outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}
