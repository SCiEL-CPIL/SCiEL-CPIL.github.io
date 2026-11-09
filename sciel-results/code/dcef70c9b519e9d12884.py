"""XOR circuit synthesis with heap-based priority queue and efficient cost estimation."""
import time
import heapq

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

    def estimate_cost(target):
        """Estimate the minimum number of gates needed to build target from existing wires."""
        min_pc = float('inf')
        for a in val_to_wires:
            diff = target ^ a
            if diff == 0:
                return 0
            pc = popcount(diff)
            if pc < min_pc:
                min_pc = pc
                if pc == 1:
                    break
        return 1 + min_pc if min_pc < float('inf') else popcount(target)

    def find_one_gate_pair(target):
        """Find a pair (a_wire, b_wire) such that a XOR b = target, both available."""
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                if a_wire != b_wire or target == 0:
                    return a_wire, b_wire
        return None

    def build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        """
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n

        if target in val_to_wires:
            return val_to_wires[target]

        if depth > 32:
            return build_from_inputs(target)

        # Find best helper
        best_a = None
        best_a_wire = None
        min_pc = float('inf')
        best_diff = None

        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                return a_wire
            pc = popcount(diff)
            if pc < min_pc:
                min_pc = pc
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                if pc == 1:
                    break

        if best_a is None:
            return build_from_inputs(target)

        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            return best_a_wire

        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire

        return new_wire

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Priority queue: (estimated_cost, row_index)
    pq = []
    pq_counter = 0  # For tie-breaking

    # Initialize priority queue with all rows
    for idx in range(m):
        target = rows[idx]
        if target == 0:
            row_to_wire[idx] = n
            remaining.discard(idx)
        elif target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            remaining.discard(idx)
        else:
            cost = estimate_cost(target)
            heapq.heappush(pq, (cost, pq_counter, idx))
            pq_counter += 1

    # Process rows in order of increasing estimated cost
    while pq and remaining:
        if time.time() > deadline:
            break

        cost, _, idx = heapq.heappop(pq)

        # Skip if already processed
        if idx not in remaining:
            continue

        target = rows[idx]

        # Re-check if target is now available
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            remaining.discard(idx)
            continue

        # Check for one-gate completion
        pair = find_one_gate_pair(target)
        if pair is not None:
            a_wire, b_wire = pair
            gates.append([a_wire, b_wire])
            row_to_wire[idx] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
            remaining.discard(idx)

            # Update costs of remaining rows (they may now be cheaper)
            for ridx in remaining:
                new_cost = estimate_cost(rows[ridx])
                if new_cost < cost:  # Only push if potentially better
                    heapq.heappush(pq, (new_cost, pq_counter, ridx))
                    pq_counter += 1
            continue

        # Build recursively
        row_to_wire[idx] = build_recursive(target)
        remaining.discard(idx)

        # Update costs of remaining rows
        for ridx in remaining:
            new_cost = estimate_cost(rows[ridx])
            if new_cost < cost:
                heapq.heappush(pq, (new_cost, pq_counter, ridx))
                pq_counter += 1

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
