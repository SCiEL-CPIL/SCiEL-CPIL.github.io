"""XOR-circuit synthesis via MST-inspired shared-subexpression with balanced final tree."""
import time
import numpy as np
from collections import defaultdict

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

    # Track depth per wire for tie-breaking
    wire_depth = [0] * (n_inputs + 1)

    def compute_cost():
        cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                cost += len(wires) - 1
        return cost

    # === MST-inspired approach: compute edge weights from row frequencies ===
    # Build frequency map: for each row, count which pairs of input wires co-occur
    # Then use Kruskal to select high-value shared subexpressions

    # First pass: greedy with improved scoring using frequency-weighted benefits
    max_iter = 20000
    iter_count = 0
    no_improvement = 0

    while iter_count < max_iter:
        if time.time() > deadline - 5.0:
            break
        iter_count += 1

        current_cost = compute_cost()

        # Find co-occurring pairs and their frequencies
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

        if not pair_rows:
            break

        # MST-inspired scoring: prioritize edges that create the most sharing
        # with a smooth surrogate for diminishing returns
        best_score = -1e9
        best_pair = None
        best_new_mask = None
        best_rows = None
        best_existing = -1
        best_depth = 999

        for (a, b), rows in pair_rows.items():
            if len(rows) < 2:
                continue

            new_mask = wire_mask[a] ^ wire_mask[b]
            if new_mask == 0:
                continue

            # Direct benefit: number of rows where this pair appears
            direct_benefit = len(rows)

            # Smooth surrogate for transitive benefit:
            # Use log-sum-exp style soft counting over potential future sharing
            transitive_bonus = 0.0
            for i in rows:
                wires_i = row_wires[i]
                n_other = len(wires_i) - 2
                if n_other > 0:
                    # Smooth diminishing returns
                    transitive_bonus += 0.5 * n_other / (1.0 + 0.2 * n_other)

            if new_mask in mask_to_wire:
                existing = mask_to_wire[new_mask]
                score = direct_benefit + transitive_bonus
                depth = wire_depth[existing]
            else:
                existing = -1
                popcnt = bin(new_mask).count('1')
                # Cost of creating the wire: 1 gate + penalty for complexity
                score = direct_benefit + transitive_bonus - 1.0 - 0.02 * popcnt
                depth = max(wire_depth[a], wire_depth[b]) + 1

            # Tie-break: prefer lower depth, then higher direct benefit
            if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and depth < best_depth):
                best_score = score
                best_pair = (a, b)
                best_new_mask = new_mask
                best_rows = rows
                best_existing = existing
                best_depth = depth

        if best_pair is None or best_score <= 0:
            break

        a, b = best_pair

        if best_existing >= 0:
            new_wire = best_existing
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            wire_depth.append(max(wire_depth[a], wire_depth[b]) + 1)
            gates.append([a, b])

        # Update affected rows
        for i in best_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                if new_wire in wires:
                    wires.remove(new_wire)
                else:
                    wires.append(new_wire)

        new_cost = compute_cost()

        if new_cost < current_cost:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 100:
                break

    # === Polish pass 1: replace pairs with existing wires ===
    if time.time() < deadline - 5.0:
        improved = True
        while improved:
            improved = False
            for i in range(n_rows):
                wires = row_wires[i]
                if len(wires) < 2:
                    continue
                n_w = len(wires)
                for x in range(n_w):
                    for y in range(x + 1, n_w):
                        a, b = wires[x], wires[y]
                        if a > b:
                            a, b = b, a
                        new_mask = wire_mask[a] ^ wire_mask[b]
                        if new_mask == 0:
                            continue
                        if new_mask in mask_to_wire:
                            new_wire = mask_to_wire[new_mask]
                            if new_wire not in wires:
                                wires.remove(a)
                                wires.remove(b)
                                wires.append(new_wire)
                                improved = True
                                break
                    if improved:
                        break

    # === Polish pass 2: create new wires if they reduce total cost ===
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            current_cost = compute_cost()

            # Find all possible pairs across all rows
            pair_counts = defaultdict(int)
            pair_example = {}
            for i, wires in enumerate(row_wires):
                n_w = len(wires)
                if n_w < 2:
                    continue
                for x in range(n_w):
                    for y in range(x + 1, n_w):
                        a, b = wires[x], wires[y]
                        if a > b:
                            a, b = b, a
                        key = (a, b)
                        pair_counts[key] += 1
                        if key not in pair_example:
                            pair_example[key] = (i, a, b)

            best_gain = 0
            best_key = None
            best_new_mask = None
            best_new_wire = -1

            for (a, b), count in pair_counts.items():
                if count < 2:
                    continue
                new_mask = wire_mask[a] ^ wire_mask[b]
                if new_mask == 0:
                    continue

                if new_mask in mask_to_wire:
                    continue  # Already handled in polish pass 1

                popcnt = bin(new_mask).count('1')
                # Gain: save (count - 1) gates from sharing, cost 1 new gate
                gain = count - 1 - 1.0 - 0.01 * popcnt
                if gain > best_gain:
                    best_gain = gain
                    best_key = (a, b)
                    best_new_mask = new_mask
                    best_new_wire = next_wire

            if best_key is None or best_gain <= 0:
                break

            a, b = best_key
            new_wire = best_new_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            wire_depth.append(max(wire_depth[a], wire_depth[b]) + 1)
            gates.append([a, b])

            # Update all rows containing this pair
            for i, wires in enumerate(row_wires):
                if a in wires and b in wires:
                    wires.remove(a)
                    wires.remove(b)
                    wires.append(new_wire)

            new_cost = compute_cost()
            if new_cost < current_cost:
                improved = True
            else:
                # Revert
                gates.pop()
                wire_mask.pop()
                del mask_to_wire[best_new_mask]
                wire_depth.pop()
                next_wire -= 1
                for i, wires in enumerate(row_wires):
                    if new_wire in wires:
                        wires.remove(new_wire)
                        wires.append(a)
                        wires.append(b)
                break

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
