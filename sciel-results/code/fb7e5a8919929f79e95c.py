"""XOR circuit synthesis with basis-aware ordering and refined cost-minimizing recursive builder."""
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

    # Identify basis rows using Gaussian Elimination for ordering
    basis_indices = []
    non_basis_indices = []
    processed_rows = []  # list of (reduced_row, pivot_bit)

    for i in range(m):
        r = rows[i]
        if r == 0:
            non_basis_indices.append(i)
            continue

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

    # Sort basis indices by popcount (sparse first)
    def popcount(x):
        return bin(x).count('1')

    basis_indices.sort(key=lambda i: popcount(rows[i]))
    non_basis_indices.sort(key=lambda i: popcount(rows[i]))

    # Combined processing order: Basis first, then non-basis
    processing_order = basis_indices + non_basis_indices

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

    # Memoization for recursive builder to avoid redundant work
    memo = {}  # target -> (wire, gates_added)

    def build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        Uses memoization and refined cost estimation.
        """
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n

        if target in val_to_wires:
            return val_to_wires[target]

        if target in memo:
            return memo[target]

        if depth > 32:
            # Fallback: build from inputs directly
            return build_from_inputs(target)

        # Find 'a' in val_to_wires that minimizes the estimated cost of (target ^ a)
        # Refined cost: if diff is in val_to_wires, cost = 1; else use popcount as proxy
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        best_diff = None

        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                # target == a, already available
                memo[target] = a_wire
                return a_wire

            # Refined cost estimation
            if diff in val_to_wires:
                cost = 1  # One gate: a XOR diff
            else:
                # Estimate: 1 + popcount(diff) as a rough upper bound
                # But if popcount is small, the actual cost might be lower
                pc = popcount(diff)
                if pc <= 2:
                    cost = 2  # Likely 1-2 gates
                elif pc <= 4:
                    cost = 3
                else:
                    cost = 1 + pc // 2  # Conservative estimate

            if cost < min_cost:
                min_cost = cost
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                if cost == 1:
                    # Optimal: diff is already available
                    break

        if best_a is None:
            # Should not happen, but fallback
            return build_from_inputs(target)

        # Recursively build the diff
        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            memo[target] = best_a_wire
            return best_a_wire

        # Create gate: best_a_wire XOR diff_wire
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire
        memo[target] = new_wire

        return new_wire

    def build_from_inputs(target):
        """Fallback: build target directly from input wires using a chain."""
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]
        if target in memo:
            return memo[target]

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
        memo[target] = wire
        return wire

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Iteratively pick the row with minimum additional cost
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

            if target in memo:
                row_to_wire[idx] = memo[target]
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

            # Estimate recursive cost using refined model
            est_cost = float('inf')
            for a, a_wire in val_to_wires.items():
                diff = target ^ a
                if diff == 0:
                    est_cost = 0
                    break
                if diff in val_to_wires:
                    est = 1
                else:
                    pc = popcount(diff)
                    if pc <= 2:
                        est = 2
                    elif pc <= 4:
                        est = 3
                    else:
                        est = 1 + pc // 2
                if est < est_cost:
                    est_cost = est

            if est_cost < best_cost:
                best_cost = est_cost
                best_row = idx
                best_type = 'recursive'
                best_a_wire = None
                best_b_wire = None

        if best_row is None:
            break

        target = rows[best_row]

        if best_type == 'one_gate':
            gates.append([best_a_wire, best_b_wire])
            row_to_wire[best_row] = next_wire
            val_to_wires[target] = next_wire
            memo[target] = next_wire
            next_wire += 1
        elif best_type == 'recursive':
            row_to_wire[best_row] = build_recursive(target)
        else:
            row_to_wire[best_row] = n

        remaining.discard(best_row)

    # Ensure all rows have a wire assigned
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wires:
                row_to_wire[i] = val_to_wires[rows[i]]
            elif rows[i] in memo:
                row_to_wire[i] = memo[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
