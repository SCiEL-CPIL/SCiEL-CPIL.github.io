# EVOLVE-BLOCK-START
import time
import sys
import heapq
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 48.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    gate_masks = []  # mask for each gate index (wire - (n+1))

    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def deadline_reached():
        return elapsed() > DEADLINE

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def add_gate_masked(a, b, new_mask):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        gate_masks.append(new_mask)
        return w

    # Recursive builder with memoization and deadline guard.
    # It builds a wire for a given mask by splitting into two submasks.
    # We prefer splits that are "balanced" and also try to hit memo.
    def build_wire(mask):
        if deadline_reached():
            raise TimeoutError
        if mask == 0:
            return n
        if mask in memo:
            return memo[mask]

        # Single bit
        if mask & (mask - 1) == 0:
            j = mask.bit_length() - 1
            memo[mask] = j
            return j

        # Try to find a good split:
        # We want to split mask into sub1, sub2 such that sub1 ^ sub2 == mask.
        # Heuristic: split by a bit that appears in many rows? For now, split by highest bit.
        # Better: try a few candidate pivots and pick the one that leads to memo hits or balanced sizes.
        # For speed, we do a deterministic split first: highest bit.
        # Then if that fails to be efficient, we can randomize.

        # Let's try a few split strategies:
        # 1. Highest bit
        # 2. Lowest bit
        # 3. Middle bit (by index)
        # 4. Random bit

        candidates = []

        # Highest bit
        hb = mask.bit_length() - 1
        candidates.append(hb)

        # Lowest bit
        lb = (mask & -mask).bit_length() - 1
        if lb != hb:
            candidates.append(lb)

        # Middle bit (by index, not by popcount)
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) >= 3:
            mid_bit = bits[len(bits) // 2]
            if mid_bit not in candidates:
                candidates.append(mid_bit)

        # Add a random bit if we have time
        if not deadline_reached() and len(bits) > 2:
            rand_bit = random.choice(bits)
            if rand_bit not in candidates:
                candidates.append(rand_bit)

        best_wire = None
        best_cost = float('inf')

        for pivot_bit in candidates:
            if deadline_reached():
                break
            sub1_mask = mask & ~(1 << pivot_bit)
            sub2_mask = 1 << pivot_bit

            # Check if submasks are in memo
            cost = 0
            if sub1_mask not in memo:
                cost += sub1_mask.bit_count()  # rough estimate
            if sub2_mask not in memo:
                cost += 1

            # Prefer splits with lower estimated cost
            if cost < best_cost:
                best_cost = cost
                best_pivot = pivot_bit
                best_sub1 = sub1_mask
                best_sub2 = sub2_mask

        if best_wire is None:
            # Fallback: use highest bit
            pivot_bit = hb
            best_sub1 = mask & ~(1 << pivot_bit)
            best_sub2 = 1 << pivot_bit

        # Recursively build submasks
        if best_sub1 in memo:
            w1 = memo[best_sub1]
        else:
            w1 = build_wire(best_sub1)

        if best_sub2 in memo:
            w2 = memo[best_sub2]
        else:
            w2 = build_wire(best_sub2)

        # Create gate
        w = add_gate_masked(w1, w2, mask)
        memo[mask] = w
        return w

    # Phase 1: Build all row wires using the recursive builder.
    # Sort rows by popcount to build simpler ones first (may help memo).
    row_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count())

    outputs = [None] * m
    try:
        for i in row_indices:
            if deadline_reached():
                # If we ran out of time, try to fill remaining with a simple fallback
                # but this should not happen often.
                break
            mask = row_masks[i]
            if mask == 0:
                outputs[i] = n
            else:
                outputs[i] = build_wire(mask)
    except TimeoutError:
        pass

    # Fill any missing outputs with a simple fallback (should be rare)
    for i in range(m):
        if outputs[i] is None:
            mask = row_masks[i]
            if mask == 0:
                outputs[i] = n
            else:
                # Simple fallback: build from bits
                bits = [j for j in range(n) if (mask >> j) & 1]
                if not bits:
                    outputs[i] = n
                elif len(bits) == 1:
                    outputs[i] = bits[0]
                else:
                    # Pairwise XOR
                    wires = bits[:]
                    while len(wires) > 1:
                        new_wires = []
                        for k in range(0, len(wires), 2):
                            if k + 1 < len(wires):
                                w = add_gate(wires[k], wires[k+1])
                                new_wires.append(w)
                            else:
                                new_wires.append(wires[k])
                        wires = new_wires
                    outputs[i] = wires[0]

    # Phase 2: Local search to reduce gate count (optional, if time permits)
    # We can try to replace gates with equivalent ones that are already in memo.
    # For now, we skip this to keep the code simple and fast.

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
