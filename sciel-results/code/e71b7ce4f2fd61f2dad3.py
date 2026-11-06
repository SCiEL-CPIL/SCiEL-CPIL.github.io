import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to bitmasks for faster subset checks
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        rows.append(mask)

    # Wire assignment
    next_wire = n + 1
    gates = []

    # Memo: bitmask -> (wire_id, depth)
    memo = {}
    for j in range(n):
        memo[1 << j] = (j, 0)
    memo[0] = (n, 0)

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b, d_a, d_b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        depth = max(d_a, d_b) + 1
        return w, depth

    def balanced_tree_xor(elems, memo):
        """Build XOR tree from list of single-bit masks using balanced tree."""
        if not elems:
            return n, 0
        if len(elems) == 1:
            j = elems[0].bit_length() - 1
            return j, 0

        wires_depths = []
        for e in elems:
            j = e.bit_length() - 1
            wires_depths.append((j, 0))

        while len(wires_depths) > 1:
            new_wires = []
            for i in range(0, len(wires_depths), 2):
                if i + 1 < len(wires_depths):
                    w1, d1 = wires_depths[i]
                    w2, d2 = wires_depths[i + 1]
                    w, d = add_gate(w1, w2, d1, d2)
                    new_wires.append((w, d))
                else:
                    new_wires.append(wires_depths[i])
            wires_depths = new_wires
        return wires_depths[0]

    def compute_set(mask):
        if mask in memo:
            return memo[mask]

        # If deadline exceeded, fallback to balanced tree to ensure termination
        if elapsed() > DEADLINE:
            if mask == 0:
                memo[mask] = (n, 0)
                return memo[mask]
            if (mask & (mask - 1)) == 0:
                j = mask.bit_length() - 1
                memo[mask] = (j, 0)
                return memo[mask]

            elems = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                elems.append(lsb.bit_length() - 1)
                temp_mask ^= lsb

            w, d = balanced_tree_xor(elems, memo)
            memo[mask] = (w, d)
            return (w, d)

        # Phase 1: Find split mask = A | B where both A, B are in memo
        # This gives cost 1 gate.
        for A, (wire_A, depth_A) in memo.items():
            if (A & mask) != A:
                continue
            B = mask ^ A
            if B in memo:
                wire_B, depth_B = memo[B]
                w, d = add_gate(wire_A, wire_B, depth_A, depth_B)
                memo[mask] = (w, d)
                return (w, d)

        # Phase 2: Find largest A in memo that is subset of mask
        # Then compute B = mask ^ A recursively
        best_A_size = -1
        best_A = None
        best_wire_A = None
        best_depth_A = None

        for A, (wire_A, depth_A) in memo.items():
            if (A & mask) != A:
                continue
            size_A = A.bit_count()
            if size_A > best_A_size:
                best_A_size = size_A
                best_A = A
                best_wire_A = wire_A
                best_depth_A = depth_A

        if best_A is not None:
            B = mask ^ best_A
            w_B, d_B = compute_set(B)
            w, d = add_gate(best_wire_A, w_B, best_depth_A, d_B)
            memo[mask] = (w, d)
            return (w, d)

        # Fallback: Balanced tree
        if mask == 0:
            memo[mask] = (n, 0)
            return memo[mask]
        if (mask & (mask - 1)) == 0:
            j = mask.bit_length() - 1
            memo[mask] = (j, 0)
            return memo[mask]

        elems = []
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            elems.append(lsb.bit_length() - 1)
            temp_mask ^= lsb

        w, d = balanced_tree_xor(elems, memo)
        memo[mask] = (w, d)
        return (w, d)

    outputs = []
    sorted_indices = sorted(range(m), key=lambda i: rows[i].bit_count())

    for i in sorted_indices:
        if elapsed() > DEADLINE:
            # Fallback for remaining rows
            for j in range(len(sorted_indices)):
                if j >= len(outputs):
                    w, d = compute_set(rows[sorted_indices[j]])
                    outputs.append(w)
            break

        w, d = compute_set(rows[i])
        outputs.append(w)

    final_outputs = [0] * m
    for idx, orig_idx in enumerate(sorted_indices):
        final_outputs[orig_idx] = outputs[idx]

    return {'gates': gates, 'outputs': final_outputs}
