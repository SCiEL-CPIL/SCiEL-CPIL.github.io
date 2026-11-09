"""XOR circuit synthesis with adaptive ordering, CSE, and pairwise-lookup acceleration."""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin before 90s kill

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

    # Memoization for build_recursive to avoid recomputation
    build_cache = {}

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

        if target in build_cache:
            return build_cache[target]

        if depth > 32:
            # Fallback: build from inputs directly
            result = build_from_inputs(target)
            build_cache[target] = result
            return result

        # Find 'a' in val_to_wires that minimizes the estimated cost of (target ^ a)
        # Limit search to top candidates to reduce O(W) scan
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        best_diff = None

        # First pass: check for exact match or single-bit diff (fast wins)
        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                build_cache[target] = a_wire
                return a_wire
            pc = popcount(diff)
            if pc == 1:
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                min_cost = 1
                break

        # If no single-bit diff found, do a more careful search
        if best_a is None:
            # Sort candidates by popcount of diff to find good helpers quickly
            # But to avoid O(W log W), just scan and track best
            for a, a_wire in val_to_wires.items():
                diff = target ^ a
                if diff == 0:
                    build_cache[target] = a_wire
                    return a_wire
                pc = popcount(diff)
                if pc < min_cost:
                    min_cost = pc
                    best_a = a
                    best_a_wire = a_wire
                    best_diff = diff
                    if pc == 1:
                        break

        if best_a is None:
            result = build_from_inputs(target)
            build_cache[target] = result
            return result

        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            build_cache[target] = best_a_wire
            return best_a_wire

        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire
        build_cache[target] = new_wire

        return new_wire

    def find_one_gate_pair(target):
        """
        Find a pair (a_wire, b_wire) such that a_wire XOR b_wire = target,
        both already in val_to_wires, and a_wire != b_wire.
        Returns (a_wire, b_wire) or None.
        Uses O(|val_to_wires|) but with early exit on first valid pair.
        """
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                if a_wire != b_wire:
                    return (a_wire, b_wire)
        return None

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Pre-compute popcount for each row for cost estimation
    row_popcount = [popcount(rows[i]) for i in range(m)]

    # Use a priority queue for cost-based scheduling
    # Each entry: (estimated_cost, row_index)
    pq = []
    
    def estimate_cost(target):
        """Estimate the cost to build target from current wires."""
        if target == 0:
            return 0
        if target in val_to_wires:
            return 0
        
        best_est = float('inf')
        for a in val_to_wires.keys():
            diff = target ^ a
            if diff == 0:
                return 0
            pc = popcount(diff)
            est = 1 + pc
            if est < best_est:
                best_est = est
        return best_est

    # Initialize priority queue with all remaining rows
    for idx in range(m):
        target = rows[idx]
        if target == 0:
            row_to_wire[idx] = n
            remaining.discard(idx)
            continue
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            remaining.discard(idx)
            continue
        est = estimate_cost(target)
        heapq.heappush(pq, (est, idx))

    # Iteratively pick the row with minimum estimated cost
    while remaining:
        if time.time() > deadline:
            break

        # Pop the cheapest row from the priority queue
        best_est, best_idx = heapq.heappop(pq)
        
        if best_idx not in remaining:
            continue  # Already processed
        
        target = rows[best_idx]
        
        # Check if it's now available (may have been completed as a side effect)
        if target in val_to_wires:
            row_to_wire[best_idx] = val_to_wires[target]
            remaining.discard(best_idx)
            continue

        if target == 0:
            row_to_wire[best_idx] = n
            remaining.discard(best_idx)
            continue

        # Check for one-gate completion first
        pair = find_one_gate_pair(target)
        if pair is not None:
            a_wire, b_wire = pair
            gates.append([a_wire, b_wire])
            row_to_wire[best_idx] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
            remaining.discard(best_idx)
            
            # After adding a new wire, some other rows may now be one-gate completable
            # Re-estimate costs for remaining rows that might benefit
            for other_idx in list(remaining):
                other_target = rows[other_idx]
                if other_target in val_to_wires:
                    row_to_wire[other_idx] = val_to_wires[other_target]
                    remaining.discard(other_idx)
                    heapq.heappush(pq, (0, other_idx))  # Mark as done
                else:
                    # Check if one-gate now
                    other_pair = find_one_gate_pair(other_target)
                    if other_pair is not None:
                        heapq.heappush(pq, (1, other_idx))
                    else:
                        new_est = estimate_cost(other_target)
                        heapq.heappush(pq, (new_est, other_idx))
            continue

        # No one-gate completion; use recursive builder
        row_to_wire[best_idx] = build_recursive(target)
        remaining.discard(best_idx)
        
        # After building, re-estimate costs for remaining rows
        for other_idx in list(remaining):
            other_target = rows[other_idx]
            if other_target in val_to_wires:
                row_to_wire[other_idx] = val_to_wires[other_target]
                remaining.discard(other_idx)
                heapq.heappush(pq, (0, other_idx))
            else:
                new_est = estimate_cost(other_target)
                heapq.heappush(pq, (new_est, other_idx))

    # Ensure all rows have a wire assigned
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wires:
                row_to_wire[i] = val_to_wires[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    # Polish pass: CSE cleanup - check if any gate's output value is already
    # available via another wire, and if so, replace references and remove the gate
    if len(gates) > 0 and time.time() < deadline - 3.0:
        # Build wire-to-value map
        wire_to_val = {}
        for j in range(n):
            wire_to_val[j] = 1 << j
        wire_to_val[n] = 0  # constant zero
        for k in range(len(gates)):
            a_wire = gates[k][0]
            b_wire = gates[k][1]
            out_wire = n + 1 + k
            wire_to_val[out_wire] = wire_to_val.get(a_wire, 0) ^ wire_to_val.get(b_wire, 0)

        # For each gate (in reverse order to avoid index issues), check if its
        # output value is already produced by an earlier wire
        # If so, we can replace all references to this gate's output with the
        # earlier wire and remove the gate
        # This is a simple single-pass CSE; multiple passes could be done but
        # we limit to one pass for time safety

        # Build value-to-first-wire map (earliest wire that produces each value)
        val_to_first_wire = {}
        for j in range(n):
            val_to_first_wire[1 << j] = j
        val_to_first_wire[0] = n

        new_gates = []
        wire_remap = {}  # old wire -> new wire

        for k in range(len(gates)):
            out_wire = n + 1 + k
            a_wire = gates[k][0]
            b_wire = gates[k][1]
            out_val = wire_to_val[out_wire]

            # Check if this value is already available
            if out_val in val_to_first_wire:
                # Replace with existing wire
                wire_remap[out_wire] = val_to_first_wire[out_val]
                # Don't add this gate
                continue

            # Remap inputs
            new_a = wire_remap.get(a_wire, a_wire)
            new_b = wire_remap.get(b_wire, b_wire)

            # Add the gate with remapped inputs
            new_gates.append([new_a, new_b])

            # Register this wire's value
            new_out_wire = n + 1 + len(new_gates) - 1
            wire_remap[out_wire] = new_out_wire
            if out_val not in val_to_first_wire:
                val_to_first_wire[out_val] = new_out_wire

        if len(new_gates) < len(gates):
            gates = new_gates
            # Remap outputs
            for i in range(m):
                if row_to_wire[i] in wire_remap:
                    row_to_wire[i] = wire_remap[row_to_wire[i]]

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
