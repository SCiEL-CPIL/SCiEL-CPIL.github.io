"""XOR circuit synthesis with basis-aware ordering and shared sub-expression discovery."""
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0 or m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(n):
        val = 0
        for j in range(m):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Identify basis rows using Gaussian Elimination
    # Process basis rows first to establish strong available wires
    basis_indices = []
    non_basis_indices = []
    processed_rows = []  # (reduced_row, pivot_bit)
    
    for i in range(n):
        r = rows[i]
        if r == 0:
            non_basis_indices.append(i)
            continue
        
        # Reduce r by processed rows
        reduced_r = r
        for pr, pivot_bit in processed_rows:
            if (reduced_r >> pivot_bit) & 1:
                reduced_r ^= pr
        
        if reduced_r != 0:
            pivot_bit = (reduced_r & (-reduced_r)).bit_length() - 1
            processed_rows.append((r, pivot_bit))
            basis_indices.append(i)
        else:
            non_basis_indices.append(i)
    
    # Sort by popcount (sparse first) for efficient wire building
    basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    non_basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    
    processing_order = basis_indices + non_basis_indices
    
    # Wire management
    val_to_wires = {}  # bitmask -> wire_id
    for j in range(m):
        val_to_wires[1 << j] = j
    val_to_wires[0] = m  # zero wire
    
    gates = []
    next_wire = m + 1
    row_to_wire = [None] * n
    
    def popcount(x):
        return bin(x).count('1')
    
    def find_two_wire_solution(target):
        """Find two existing wires whose XOR equals target. Returns (w1, w2) or None."""
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                if a_wire != b_wire or target == 0:
                    return (a_wire, b_wire)
        return None
    
    def build_recursive(target):
        """Recursively build target using existing wires, minimizing new gates."""
        nonlocal next_wire, gates, val_to_wires
        
        if target in val_to_wires:
            return val_to_wires[target]
        if target == 0:
            return m
        
        # Find the existing wire a that minimizes popcount(target ^ a)
        best_a = None
        best_a_wire = None
        min_pc = float('inf')
        
        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            pc = popcount(diff)
            if pc < min_pc:
                min_pc = pc
                best_a = a
                best_a_wire = a_wire
                if pc == 1:
                    break
        
        if best_a is None:
            best_a = 0
            best_a_wire = m
        
        diff = target ^ best_a
        
        if diff == 0:
            return best_a_wire
        
        # Recursively build diff
        diff_wire = build_recursive(diff)
        
        # Create gate: best_a_wire XOR diff_wire
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire
        
        return new_wire
    
    # Phase 1: Process rows in priority order
    for idx in processing_order:
        if time.time() > deadline:
            break
        
        target = rows[idx]
        
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            continue
        
        # Try 1-gate solution first
        pair = find_two_wire_solution(target)
        if pair:
            w1, w2 = pair
            gates.append([w1, w2])
            row_to_wire[idx] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
        else:
            # Recursive build
            row_to_wire[idx] = build_recursive(target)
    
    # Phase 2: Shared sub-expression discovery
    # For each pair of existing wires, check if their XOR creates a new useful wire
    # that is a submask of some uncomputed row
    uncomputed_rows = [i for i in range(n) if row_to_wire[i] is None]
    
    if uncomputed_rows:
        # Collect all submasks of uncomputed rows
        target_submasks = set()
        for idx in uncomputed_rows:
            mask = rows[idx]
            sub = mask
            while sub:
                target_submasks.add(sub)
                sub = (sub - 1) & mask
        
        # Try to create new wires from XOR of existing wires
        wire_list = list(val_to_wires.items())
        new_wires_created = 0
        
        for i in range(len(wire_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(wire_list)):
                if time.time() > deadline:
                    break
                a_val, a_wire = wire_list[i]
                b_val, b_wire = wire_list[j]
                xor_val = a_val ^ b_val
                
                if xor_val == 0 or xor_val in val_to_wires:
                    continue
                
                # Only create if this submask is useful
                if xor_val in target_submasks:
                    gates.append([a_wire, b_wire])
                    new_wire = next_wire
                    next_wire += 1
                    val_to_wires[xor_val] = new_wire
                    wire_list.append((xor_val, new_wire))
                    new_wires_created += 1
    
    # Phase 3: Complete remaining rows
    for idx in range(n):
        if time.time() > deadline:
            # Emergency fallback for remaining rows
            if row_to_wire[idx] is None:
                target = rows[idx]
                if target == 0:
                    row_to_wire[idx] = m
                elif target in val_to_wires:
                    row_to_wire[idx] = val_to_wires[target]
                else:
                    # Build from scratch
                    bits = [j for j in range(m) if (target >> j) & 1]
                    if not bits:
                        row_to_wire[idx] = m
                    elif len(bits) == 1:
                        row_to_wire[idx] = bits[0]
                    else:
                        w = bits[0]
                        for j in bits[1:]:
                            gates.append([w, j])
                            w = next_wire
                            next_wire += 1
                        row_to_wire[idx] = w
            continue
        
        if row_to_wire[idx] is not None:
            continue
        
        target = rows[idx]
        
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            continue
        
        # Try 1-gate solution
        pair = find_two_wire_solution(target)
        if pair:
            w1, w2 = pair
            gates.append([w1, w2])
            row_to_wire[idx] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
        else:
            # Recursive build
            row_to_wire[idx] = build_recursive(target)
    
    # Build outputs in original order
    outputs = row_to_wire
    
    # Verify all outputs are valid
    for i in range(n):
        if outputs[i] is None:
            outputs[i] = m  # Fallback to zero
    
    return {'gates': gates, 'outputs': outputs}
