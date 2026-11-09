"""XOR circuit synthesis with adaptive ordering and cost-minimizing recursive builder."""
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Map from bitmask value to wire index
    val_to_wires = {}

    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j

    # Constant zero
    val_to_wires[0] = n

    gates = []
    next_wire = n + 1

    def popcount(x):
        return bin(x).count('1')

    def build_from_inputs(target):
        """Fallback: build target directly from input wires using a chain."""
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]

        # Build from set bits
        wire = None
        bits = target
        while bits:
            lb = bits & (-bits)
            bit_idx = lb.bit_length() - 1
            input_wire = bit_idx  # input wire for bit bit_idx
            if wire is None:
                wire = input_wire
            else:
                gates.append([wire, input_wire])
                wire = next_wire
                next_wire += 1
            bits ^= lb

        if wire is None:
            wire = n

        val_to_wires[target] = wire
        return wire

    def build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        Tries to minimize total gate count by picking the helper that
        leads to the simplest remaining subproblem.
        """
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n

        if target in val_to_wires:
            return val_to_wires[target]

        if depth > 32:
            # Fallback: build from inputs directly
            return build_from_inputs(target)

        # Find 'a' in val_to_wires that minimizes the estimated cost of (target ^ a)
        # Cost estimate: popcount of the diff (proxy for number of gates needed)
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        best_diff = None

        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                # target == a, already available
                return a_wire
            # Estimate cost: use popcount as a lower bound on gates needed
            cost = popcount(diff)
            if cost < min_cost:
                min_cost = cost
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                if cost == 1:
                    # Optimal: diff is a single input bit
                    break

        if best_a is None:
            # Should not happen, but fallback
            return build_from_inputs(target)

        # Recursively build the diff
        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            return best_a_wire

        # Create gate: best_a_wire XOR diff_wire
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire

        return new_wire

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Iteratively pick the row with minimum additional cost
    # Cost of a row: 0 if already in val_to_wires, 1 if one-gate possible, else estimated recursive cost
    while remaining:
        if time.time() > deadline:
            break

        best_row = None
        best_cost = float('inf')
        best_type = None  # 'available', 'one_gate', 'recursive'
        best_a_wire = None
        best_b_wire = None

        for idx in remaining:
            target = rows[idx]

            if target == 0:
                row_to_wire[idx] = n
                remaining.discard(idx)
                continue

            if target in val_to_wires:
                row_to_wire[idx] = val_to_wires[target]
                remaining.discard(idx)
                continue

            # Check for one-gate completion
            one_gate_found = False
            for a, a_wire in val_to_wires.items():
                b = target ^ a
                if b in val_to_wires:
                    b_wire = val_to_wires[b]
                    if a_wire == b_wire and target != 0:
                        continue
                    # One gate sufficient
                    if 1 < best_cost:
                        best_cost = 1
                        best_row = idx
                        best_type = 'one_gate'
                        best_a_wire = a_wire
                        best_b_wire = b_wire
                    one_gate_found = True
                    break

            if one_gate_found:
                continue

            # Estimate recursive cost: popcount of target (upper bound)
            # Better: try to find the best helper and estimate
            est_cost = float('inf')
            for a, a_wire in val_to_wires.items():
                diff = target ^ a
                if diff == 0:
                    est_cost = 0
                    break
                diff_pc = popcount(diff)
                # Rough estimate: 1 + cost of diff
                # Use popcount as proxy
                est = 1 + diff_pc
                if est < est_cost:
                    est_cost = est

            if est_cost < best_cost:
                best_cost = est_cost
                best_row = idx
                best_type = 'recursive'
                best_a_wire = None
                best_b_wire = None

        if best_row is None:
            # All remaining should be handled
            break

        target = rows[best_row]

        if best_type == 'one_gate':
            gates.append([best_a_wire, best_b_wire])
            row_to_wire[best_row] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
        elif best_type == 'recursive':
            row_to_wire[best_row] = build_recursive(target)
        else:
            # Should not happen
            row_to_wire[best_row] = n

        remaining.discard(best_row)

    # Ensure all rows have a wire assigned
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wires:
                row_to_wire[i] = val_to_wires[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
