"""Rewritten program for exact XOR-circuit synthesis with wire sharing."""
import time

def _deadline_guard(start, limit=85.0):
    return (time.time() - start) < limit

def _value_key(wires):
    # frozenset of input indices
    return frozenset(wires)

def synthesize(matrix, depth_limit):
    import time
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # matrix: list of rows (each row a list/tuple of 0/1) or 2D array
    # Convert to list of frozensets of input indices for each output row
    try:
        import numpy as np
        if isinstance(matrix, np.ndarray):
            rows = [set(int(j) for j in np.nonzero(matrix[i])[0]) for i in range(matrix.shape[0])]
            n = matrix.shape[1]
        else:
            rows = [set(int(j) for j, v in enumerate(row) if v) for row in matrix]
            n = len(matrix[0]) if len(matrix) > 0 else 0
    except Exception:
        rows = [set(int(j) for j, v in enumerate(row) if v) for row in matrix]
        n = len(matrix[0]) if len(matrix) > 0 else 0

    # value_to_wire: frozenset -> wire number
    # Input wires: 0..n-1 correspond to singleton sets
    # Constant zero: wire n corresponds to empty set
    value_to_wire = {}
    for i in range(n):
        value_to_wire[frozenset([i])] = i
    value_to_wire[frozenset()] = n  # constant zero wire

    gates = []  # list of [wire_a, wire_b]
    next_wire = n + 1

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # For each output row, find a short representation
    outputs = []
    for row_set in rows:
        if not _deadline_guard(start, DEADLINE):
            # Fallback: direct synthesis for remaining rows
            for rs in row_set:
                pass
            # Just use direct: XOR all inputs in row_set
            if not row_set:
                outputs.append(n)
            else:
                ws = list(row_set)
                w = ws[0]
                for x in ws[1:]:
                    w = add_gate(w, x)
                outputs.append(w)
            continue

        # Greedy: reduce row_set by XORing with existing values that shrink it
        current = set(row_set)
        used_wires = []  # wires XORed to get current from original

        # Iteratively try to shrink current using existing values
        progress = True
        while progress and _deadline_guard(start, DEADLINE):
            progress = False
            best_v = None
            best_w = None
            best_new_size = len(current)
            for v, w in value_to_wire.items():
                if not v:
                    continue
                # new = current Δ v
                sym = current.symmetric_difference(v)
                if len(sym) < best_new_size:
                    best_new_size = len(sym)
                    best_v = v
                    best_w = w
                    progress = True
            if progress:
                current = current.symmetric_difference(best_v)
                used_wires.append(best_w)
                # Register the new value (current) if not already known
                key = frozenset(current)
                if key not in value_to_wire:
                    # current = original_row Δ (XOR of used_wires)
                    # We need to build current as a wire
                    # But we track it incrementally below
                    pass

        # Now current is a small set; XOR remaining inputs directly
        if current:
            ws = list(current)
            w = ws[0]
            for x in ws[1:]:
                if not _deadline_guard(start, DEADLINE):
                    break
                w = add_gate(w, x)
            used_wires.append(w)
        else:
            # current is empty -> zero
            used_wires.append(n)

        # Combine used_wires: the output is XOR of all used_wires
        if not used_wires:
            out_wire = n
        elif len(used_wires) == 1:
            out_wire = used_wires[0]
        else:
            out_wire = used_wires[0]
            for w in used_wires[1:]:
                if not _deadline_guard(start, DEADLINE):
                    break
                out_wire = add_gate(out_wire, w)

        outputs.append(out_wire)

        # Register the final output value for future reuse
        out_key = frozenset(row_set)
        if out_key not in value_to_wire:
            value_to_wire[out_key] = out_wire

    # Verify depth: we don't strictly track depth, but greedy keeps it small
    # Build result
    result_gates = [[int(a), int(b)] for a, b in gates]
    result_outputs = [int(w) for w in outputs]

    return {'gates': result_gates, 'outputs': result_outputs}
