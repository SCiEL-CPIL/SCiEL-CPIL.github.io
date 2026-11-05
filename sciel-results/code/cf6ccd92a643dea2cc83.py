"""XOR-circuit synthesis via MST-inspired global ordering with greedy CSE."""
import time
import numpy as np
from collections import defaultdict
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs

    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Convert rows to integer bitmasks for fast XOR operations
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire representation: wire_mask[w] = bitmask of inputs that wire w represents
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire

    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Each row is a list of wire indices whose XOR gives the row value
    row_wires = []
    for i in range(n_rows):
        wires = []
        mask = row_masks[i]
        for j in range(n_cols):
            if mask & (1 << j):
                wires.append(j)
        row_wires.append(wires)

    gates = []
    next_wire = n_inputs + 1

    def compute_total_cost():
        """Compute total gate count including final assembly."""
        cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                cost += len(wires) - 1
        return cost

    def build_priority_queue():
        """
        Build a priority queue of candidate XOR operations.
        Each entry: (-score, tiebreak, a, b, new_mask)
        Score reflects net benefit of introducing this intermediate wire.
        """
        pq = []
        # Find all co-occurring pairs
        pair_rows = defaultdict(list)
        for i, wires in enumerate(row_wires):
            n_w = len(wires)
            if n_w < 2:
                continue
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a, b = wires[x], wires[y]
                    if a > b:
                        a, b = b, a
                    pair_rows[(a, b)].append(i)

        for (a, b), rows in pair_rows.items():
            new_mask = wire_mask[a] ^ wire_mask[b]
            if new_mask == 0:
                continue

            # Count how many rows actually benefit (have >1 wire after this substitution)
            benefit_count = 0
            for i in rows:
                if len(row_wires[i]) > 1:
                    benefit_count += 1

            if benefit_count < 2:
                continue

            if new_mask in mask_to_wire:
                # Already have this wire; just rewire
                score = benefit_count
            else:
                # New wire: cost 1 gate, but saves benefit_count - 1 in assembly
                # Also consider popcount as a tiebreaker (simpler vectors preferred)
                popcnt = bin(new_mask).count('1')
                score = benefit_count - 1 - 0.02 * popcnt

            if score <= 0:
                continue

            # Use negative score for max-heap; tiebreak by (a,b) for determinism
            heapq.heappush(pq, (-score, a, b, new_mask))

        return pq

    def try_apply(a, b, new_mask, pq):
        """
        Try to apply the XOR operation (a,b)->new_mask.
        Returns True if applied, False if not beneficial or invalid.
        """
        nonlocal next_wire, gates, wire_mask, mask_to_wire, row_wires

        if new_mask == 0:
            return False

        # Check if both a and b still appear together in some row
        # Find rows where both a and b are present
        affected_rows = []
        for i, wires in enumerate(row_wires):
            if a in wires and b in wires:
                affected_rows.append(i)

        if len(affected_rows) < 2:
            return False

        # Recompute benefit
        benefit_count = sum(1 for i in affected_rows if len(row_wires[i]) > 1)
        if benefit_count < 2:
            return False

        old_cost = compute_total_cost()

        if new_mask in mask_to_wire:
            new_wire = mask_to_wire[new_mask]
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(new_mask)
            mask_to_wire[new_mask] = new_wire
            gates.append([a, b])

        # Update affected rows
        for i in affected_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                if new_wire not in wires:
                    wires.append(new_wire)

        new_cost = compute_total_cost()

        if new_cost >= old_cost:
            # Rollback
            if new_mask not in mask_to_wire or mask_to_wire[new_mask] != new_wire or \
               (new_wire >= n_inputs + 1 and gates and gates[-1] == [a, b]):
                # We created a new wire, remove it
                if new_wire >= n_inputs + 1:
                    if gates and gates[-1] == [a, b]:
                        gates.pop()
                    del wire_mask[new_wire]
                    del mask_to_wire[new_mask]
                    next_wire = new_wire
            # Restore rows
            for i in affected_rows:
                wires = row_wires[i]
                if new_wire in wires:
                    wires.remove(new_wire)
                    wires.append(a)
                    wires.append(b)
            return False

        return True

    # Main loop: process candidates in priority order
    max_iter = 50000
    iter_count = 0
    no_improvement = 0
    pq = build_priority_queue()

    while iter_count < max_iter:
        if time.time() > deadline - 5.0:
            break
        iter_count += 1

        if not pq:
            break

        # Pop the best candidate
        neg_score, a, b, new_mask = heapq.heappop(pq)
        score = -neg_score

        if score <= 0:
            break

        if try_apply(a, b, new_mask, pq):
            no_improvement = 0
            # Rebuild priority queue periodically to capture new opportunities
            if iter_count % 100 == 0 or len(pq) < 10:
                pq = build_priority_queue()
        else:
            no_improvement += 1
            if no_improvement > 200:
                # Try rebuilding the PQ once more
                pq = build_priority_queue()
                if not pq:
                    break
                no_improvement = 0

    # Final output assembly with balanced binary tree to control depth
    outputs = []
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            # Build balanced XOR tree
            current_level = list(wires)
            while len(current_level) > 1:
                next_level = []
                for k in range(0, len(current_level), 2):
                    if k + 1 < len(current_level):
                        w = next_wire
                        next_wire += 1
                        gates.append([current_level[k], current_level[k + 1]])
                        next_level.append(w)
                    else:
                        next_level.append(current_level[k])
                current_level = next_level
            outputs.append(current_level[0])

    return {'gates': gates, 'outputs': outputs}
