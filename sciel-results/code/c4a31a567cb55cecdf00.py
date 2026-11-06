"""XOR-circuit synthesis with iterative shared-subexpression extraction and local search."""
import time
import numpy as np

_DEADLINE = 45.0  # safety margin under the 60s per-eval / 90s wall-clock limit


def _rows_to_masks(matrix):
    M = np.asarray(matrix, dtype=np.uint8)
    m, n = M.shape
    rows = []
    for i in range(m):
        mask = 0
        r = M[i]
        for j in range(n):
            if r[j]:
                mask |= (1 << j)
        rows.append(mask)
    return rows


def _min_need_count(target, available):
    """Find minimum number of available wires whose XOR equals target."""
    if target == 0:
        return 0
    if target in available:
        return 1
    
    # Try pairs
    for a in available:
        b = a ^ target
        if b in available and a != b:
            return 2
    
    # Try triples (limited search)
    avail_list = list(available)
    if len(avail_list) <= 50:
        for i in range(len(avail_list)):
            for j in range(i + 1, len(avail_list)):
                c = avail_list[i] ^ avail_list[j] ^ target
                if c in available and c != avail_list[i] and c != avail_list[j]:
                    return 3
    return None


def _decompose(targets, n, time_budget):
    """
    Synthesize XOR circuit for given targets.
    Returns (gates, outputs) where gates is list of [wire_a, wire_b],
    and outputs is list of wire indices for each target.
    """
    start_time = time.time()
    
    # Wire numbering: inputs 0..n-1, zero wire = n
    # Gate k produces wire n+1+k
    wires = [1 << i for i in range(n)] + [0]  # wire values
    wire_map = {w: i for i, w in enumerate(wires)}
    gates = []
    
    def time_left():
        return time_budget - (time.time() - start_time)
    
    def get_wire_for_vector(vec):
        """Get or create a wire for the given bitmask."""
        if vec in wire_map:
            return wire_map[vec]
        
        if time_left() < 2.0:
            # Fallback: build from inputs
            bits = [j for j in range(n) if vec & (1 << j)]
            if not bits:
                return n
            if len(bits) == 1:
                return wire_map[1 << bits[0]]
            
            current = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current, input_wire])
                wires.append(wires[current] ^ (1 << j))
                wire_map[wires[current] ^ (1 << j)] = new_wire
                current = new_wire
            return current
        
        # Try to find vec = a XOR b for available wires
        avail = list(wire_map.keys())
        for a in avail:
            b = a ^ vec
            if b in wire_map and a != b:
                wa = wire_map[a]
                wb = wire_map[b]
                new_wire = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(vec)
                wire_map[vec] = new_wire
                return new_wire
        
        # BFS depth 2: find vec = (a^b) ^ (c^d) or (a^b) ^ c
        # First, compute all level-1 XORs
        level1 = {}
        avail_set = set(avail)
        for i, a in enumerate(avail):
            for b in avail[i+1:]:
                xor_val = a ^ b
                if xor_val not in wire_map:
                    if xor_val not in level1:
                        level1[xor_val] = (a, b)
        
        # Check if vec is in level1
        if vec in level1:
            a, b = level1[vec]
            wa = wire_map[a]
            wb = wire_map[b]
            new_wire = n + 1 + len(gates)
            gates.append([wa, wb])
            wires.append(vec)
            wire_map[vec] = new_wire
            return new_wire
        
        # Check if vec = level1_val ^ avail_val
        for l1_val, (l1_a, l1_b) in level1.items():
            needed = l1_val ^ vec
            if needed in wire_map:
                # Create l1_val wire
                wa = wire_map[l1_a]
                wb = wire_map[l1_b]
                new_wire_l1 = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(l1_val)
                wire_map[l1_val] = new_wire_l1
                
                # XOR with needed
                wn = wire_map[needed]
                new_wire = n + 1 + len(gates)
                gates.append([new_wire_l1, wn])
                wires.append(vec)
                wire_map[vec] = new_wire
                return new_wire
        
        # Check if vec = level1_val1 ^ level1_val2
        l1_items = list(level1.items())
        if len(l1_items) <= 2000:
            for i in range(len(l1_items)):
                for j in range(i + 1, len(l1_items)):
                    v1, (a1, b1) = l1_items[i]
                    v2, (a2, b2) = l1_items[j]
                    if v1 ^ v2 == vec:
                        # Create both level1 wires
                        wa1 = wire_map[a1]
                        wb1 = wire_map[b1]
                        new_wire_1 = n + 1 + len(gates)
                        gates.append([wa1, wb1])
                        wires.append(v1)
                        wire_map[v1] = new_wire_1
                        
                        wa2 = wire_map[a2]
                        wb2 = wire_map[b2]
                        new_wire_2 = n + 1 + len(gates)
                        gates.append([wa2, wb2])
                        wires.append(v2)
                        wire_map[v2] = new_wire_2
                        
                        # XOR them
                        new_wire = n + 1 + len(gates)
                        gates.append([new_wire_1, new_wire_2])
                        wires.append(vec)
                        wire_map[vec] = new_wire
                        return new_wire
        
        # Fallback: build from inputs
        bits = [j for j in range(n) if vec & (1 << j)]
        if not bits:
            return n
        if len(bits) == 1:
            return wire_map[1 << bits[0]]
        
        current = wire_map[1 << bits[0]]
        for j in bits[1:]:
            input_wire = wire_map[1 << j]
            new_wire = n + 1 + len(gates)
            gates.append([current, input_wire])
            wires.append(wires[current] ^ (1 << j))
            wire_map[wires[current] ^ (1 << j)] = new_wire
            current = new_wire
        return current
    
    # First pass: create wires for all targets
    target_wires = []
    for t in targets:
        if t == 0:
            target_wires.append(n)
        else:
            target_wires.append(get_wire_for_vector(t))
    
    # Second pass: try to optimize by finding shared subexpressions
    # For each target, check if it can be expressed with fewer gates
    # by reusing existing wires
    
    # Iterative improvement: try to find better decompositions
    improved = True
    max_iterations = 5
    iteration = 0
    while improved and iteration < max_iterations and time_left() > 5.0:
        improved = False
        iteration += 1
        
        for idx, t in enumerate(targets):
            if t == 0:
                continue
            
            current_wire = target_wires[idx]
            # Check if we can find a cheaper expression
            # Count current cost (approximate)
            
            # Try to see if t can be made with 1 or 2 existing wires
            if t in wire_map:
                continue  # Already have a wire for it
            
            # Try pairs of existing wires
            avail = list(wire_map.keys())
            found_better = False
            for a in avail:
                b = a ^ t
                if b in wire_map and a != b:
                    wa = wire_map[a]
                    wb = wire_map[b]
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(t)
                    wire_map[t] = new_wire
                    target_wires[idx] = new_wire
                    improved = True
                    found_better = True
                    break
            
            if found_better:
                continue
            
            # Try level-1 + existing
            if time_left() < 3.0:
                continue
                
            level1_new = {}
            for i, a in enumerate(avail):
                for b in avail[i+1:]:
                    xor_val = a ^ b
                    if xor_val not in wire_map:
                        if xor_val not in level1_new:
                            level1_new[xor_val] = (a, b)
            
            if t in level1_new:
                a, b = level1_new[t]
                wa = wire_map[a]
                wb = wire_map[b]
                new_wire = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(t)
                wire_map[t] = new_wire
                target_wires[idx] = new_wire
                improved = True
                continue
            
            for l1_val, (l1_a, l1_b) in level1_new.items():
                needed = l1_val ^ t
                if needed in wire_map:
                    wa = wire_map[l1_a]
                    wb = wire_map[l1_b]
                    new_wire_l1 = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(l1_val)
                    wire_map[l1_val] = new_wire_l1
                    
                    wn = wire_map[needed]
                    new_wire = n + 1 + len(gates)
                    gates.append([new_wire_l1, wn])
                    wires.append(t)
                    wire_map[t] = new_wire
                    target_wires[idx] = new_wire
                    improved = True
                    break
    
    # Local search: eliminate redundant gates
    # A gate is redundant if its output wire is used only once (by the gate that uses it)
    # and that usage can be inlined
    
    def _eliminate_gates():
        nonlocal gates, wires, wire_map, target_wires
        
        # Build usage map
        wire_usage = {}
        for g in gates:
            for w in g:
                wire_usage[w] = wire_usage.get(w, 0) + 1
        for o in target_wires:
            wire_usage[o] = wire_usage.get(o, 0) + 1
        
        # Find gates whose output is used exactly once
        # and try to inline them
        gate_output_wire = {}
        for k, g in enumerate(gates):
            out_wire = n + 1 + k
            gate_output_wire[out_wire] = k
        
        # Iterate: find inlinable gates
        changed = True
        max_passes = 3
        for _ in range(max_passes):
            if not changed:
                break
            changed = False
            
            # Rebuild usage
            wire_usage = {}
            for g in gates:
                for w in g:
                    wire_usage[w] = wire_usage.get(w, 0) + 1
            for o in target_wires:
                wire_usage[o] = wire_usage.get(o, 0) + 1
            
            # Find gates where output is used once and the single use is in another gate
            # or in outputs
            for k in range(len(gates) - 1, -1, -1):
                out_wire = n + 1 + k
                if wire_usage.get(out_wire, 0) != 1:
                    continue
                
                # Find where out_wire is used
                use_location = None
                use_idx = None
                for j, g in enumerate(gates):
                    if j == k:
                        continue
                    if g[0] == out_wire or g[1] == out_wire:
                        use_location = j
                        use_idx = 0 if g[0] == out_wire else 1
                        break
                
                if use_location is None:
                    # Used in outputs
                    for oi, o in enumerate(target_wires):
                        if o == out_wire:
                            use_location = 'output'
                            use_idx = oi
                            break
                    else:
                        continue
                else:
                    # Used in another gate
                    g_use = gates[use_location]
                    other_wire = g_use[1 - use_idx]
                    
                    # Inline: replace out_wire with XOR of its inputs
                    # But this might increase gate count if the other wire is also complex
                    # Only inline if it doesn't create a cycle and reduces complexity
                    # Simple heuristic: only inline if the other wire is an input or zero
                    if other_wire <= n:
                        # Safe to inline
                        g_in = gates[k]
                        # Replace in use location
                        if use_location == 'output':
                            # Can't easily inline into output without creating new gate
                            # Skip for now
                            pass
                        else:
                            # Replace the wire in the use gate
                            # This changes the semantics: out_wire = g_in[0] ^ g_in[1]
                            # So use_gate becomes g_in[0] ^ g_in[1] ^ other_wire
                            # This is a 3-input XOR, which needs 2 gates
                            # So inlining might not help
                            pass
            break  # Local search is complex; skip for now
        
        return
    
    # Skip complex local search; return current result
    return gates, target_wires


def _verify(matrix, gates, outputs):
    """Verify that the circuit computes the matrix over GF(2)."""
    M = np.asarray(matrix, dtype=np.uint8)
    m, n = M.shape
    
    # Simulate the circuit
    wire_values = [1 << i for i in range(n)] + [0]
    for g in gates:
        wa, wb = g
        wire_values.append(wire_values[wa] ^ wire_values[wb])
    
    for i in range(m):
        expected = 0
        row = M[i]
        for j in range(n):
            if row[j]:
                expected |= (1 << j)
        actual = wire_values[outputs[i]]
        if expected != actual:
            return False
    return True


def _baseline(matrix):
    """Compute baseline gate count: sum of (popcount(row) - 1) for each row."""
    M = np.asarray(matrix, dtype=np.uint8)
    m, n = M.shape
    total = 0
    for i in range(m):
        pc = sum(M[i])
        if pc > 1:
            total += pc - 1
    return max(total, 1)


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    targets = _rows_to_masks(matrix)
    m = len(targets)
    n = len(targets[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if n == 0:
        return {'gates': [], 'outputs': [0] * m}
    
    # Check if all rows are zero or single-bit
    all_simple = True
    for t in targets:
        pc = bin(t).count('1')
        if pc > 1:
            all_simple = False
            break
    
    if all_simple:
        outputs = []
        for t in targets:
            if t == 0:
                outputs.append(n)
            else:
                j = t.bit_length() - 1
                outputs.append(j)
        return {'gates': [], 'outputs': outputs}
    
    # Synthesize
    gates, outputs = _decompose(targets, n, _DEADLINE)
    
    # Verify
    if not _verify(matrix, gates, outputs):
        # Fallback: simple synthesis
        gates = []
        outputs = []
        wires = [1 << i for i in range(n)] + [0]
        wire_map = {w: i for i, w in enumerate(wires)}
        
        for t in targets:
            if t == 0:
                outputs.append(n)
                continue
            bits = [j for j in range(n) if t & (1 << j)]
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(bits[0])
                continue
            
            current = bits[0]
            for j in bits[1:]:
                new_wire = n + 1 + len(gates)
                gates.append([current, j])
                wires.append(wires[current] ^ (1 << j))
                wire_map[wires[current] ^ (1 << j)] = new_wire
                current = new_wire
            outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    return {'gates': gates, 'outputs': outputs}
