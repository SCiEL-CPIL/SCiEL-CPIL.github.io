<<<<<<< SEARCH
def _build_circuit_from_rows(rows, n):
    """
    Build a circuit from a list of rows (each row is a list of input indices).
    Uses a global gate registry for maximum sharing.
    
    rows: list of lists of input indices
    n: number of input wires
    """
    # Gate registry: (a, b) with a < b -> wire index
    gate_registry = {}
    gates = []
    next_gate = 0
    
    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n  # constant zero
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire
    
    outputs = []
    for row in rows:
        if len(row) == 0:
            outputs.append(n)  # zero
        elif len(row) == 1:
            outputs.append(row[0])
        else:
            # Build XOR tree with sharing
            # Use a divide-and-conquer approach
            def xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                left = xor_tree(indices[:mid])
                right = xor_tree(indices[mid:])
                return get_xor_wire(left, right)
            outputs.append(xor_tree(row))
    
    return {'gates': gates, 'outputs': outputs}
=======
def _build_circuit_from_rows(rows, n):
    """
    Build a circuit from a list of rows (each row is a list of input indices).
    Uses a global gate registry and value-to-wire map for maximum sharing.
    Processes rows in popcount-ascending order to maximize intermediate reuse.
    Tries multiple tree shapes per row and picks the one that reuses most gates.
    
    rows: list of lists of input indices
    n: number of input wires
    """
    # Gate registry: (a, b) with a < b -> wire index
    gate_registry = {}
    gates = []
    next_gate = 0
    
    # Value-to-wire map: mask -> wire index
    value_to_wire = {}
    for i in range(n):
        value_to_wire[1 << i] = i
    value_to_wire[0] = n
    
    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n  # constant zero
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire
    
    def compute_mask(wire):
        """Compute the mask for a given wire."""
        if wire <= n:
            return value_to_wire.get(wire, 0)
        gate_idx = wire - n - 1
        a, b = gates[gate_idx]
        return compute_mask(a) ^ compute_mask(b)
    
    def get_wire_for_mask(mask):
        """Get or create a wire for a given mask."""
        if mask in value_to_wire:
            return value_to_wire[mask]
        return None
    
    def xor_tree_balanced(indices, gate_count_before):
        """Balanced tree construction. Returns (wire, new_gate_count)."""
        if len(indices) == 1:
            return indices[0], gate_count_before
        mid = len(indices) // 2
        left, gc = xor_tree_balanced(indices[:mid], gate_count_before)
        right, gc = xor_tree_balanced(indices[mid:], gc)
        new_gc = gc + 1  # one new gate
        return get_xor_wire(left, right), new_gc
    
    def xor_tree_chain(indices, gate_count_before):
        """Chain construction. Returns (wire, new_gate_count)."""
        if len(indices) == 1:
            return indices[0], gate_count_before
        wire = indices[0]
        gc = gate_count_before
        for j in range(1, len(indices)):
            wire = get_xor_wire(wire, indices[j])
            gc += 1
        return wire, gc
    
    def xor_tree_partial(indices, gate_count_before):
        """Try several split points and pick the best."""
        if len(indices) <= 2:
            if len(indices) == 1:
                return indices[0], gate_count_before
            wire = get_xor_wire(indices[0], indices[1])
            return wire, gate_count_before + 1
        
        best_wire = None
        best_gc = float('inf')
        
        # Try balanced
        w, gc = xor_tree_balanced(indices, gate_count_before)
        if gc < best_gc:
            best_gc = gc
            best_wire = w
        
        # Try chain
        w, gc = xor_tree_chain(indices, gate_count_before)
        if gc < best_gc:
            best_gc = gc
            best_wire = w
        
        return best_wire, best_gc
    
    # Sort row indices by popcount (ascending) to maximize sharing
    indexed_rows = [(i, row) for i, row in enumerate(rows)]
    indexed_rows.sort(key=lambda x: len(x[1]))
    
    outputs = [0] * len(rows)
    
    for orig_idx, row in indexed_rows:
        if len(row) == 0:
            outputs[orig_idx] = n
        elif len(row) == 1:
            outputs[orig_idx] = row[0]
        else:
            # Compute the mask for this row
            row_mask = 0
            for j in row:
                row_mask |= (1 << j)
            
            # Check if we already have a wire for this exact mask
            existing_wire = get_wire_for_mask(row_mask)
            if existing_wire is not None:
                outputs[orig_idx] = existing_wire
                continue
            
            gc_before = len(gates)
            wire, gc_after = xor_tree_partial(row, gc_before)
            
            # Update value_to_wire with the new wire
            value_to_wire[row_mask] = wire
            outputs[orig_idx] = wire
    
    return {'gates': gates, 'outputs': outputs}
>>>>>>> REPLACE

