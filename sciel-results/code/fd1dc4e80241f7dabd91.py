# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 80.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
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
    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_from_bits(bits_list):
        """
        Takes a list of input wire indices (0..n-1).
        Builds a balanced XOR tree, memoizing the mask for each new gate.
        Returns the final wire.
        """
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]

        wires = []
        masks = []
        for b in bits_list:
            wires.append(b)
            masks.append(1 << b)

        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w = wires[i]
                    a_m = masks[i]
                    b_w = wires[i + 1]
                    b_m = masks[i + 1]

                    new_m = a_m ^ b_m
                    # Check if new_m is already in memo
                    if new_m in memo:
                        # Reuse existing wire
                        new_wires.append(memo[new_m])
                        new_masks.append(new_m)
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                        new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks

        return wires[0]

    def compute_mask(mask):
        """
        Compute wire for mask using memoization and greedy nearest-neighbor search
        with an improved cost surrogate.
        """
        if mask in memo:
            return memo[mask]

        # If we hit the deadline, fallback to direct tree construction
        if elapsed() > DEADLINE - 5.0:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best M in memo to minimize estimated gate count
        # Cost(rem) = 1 if rem in memo, else popcount(rem)
        best_M = 0
        best_rem = mask
        min_score = float('inf')

        # Optimization: If mask has small popcount, the best M is likely 0 or a subset.
        # Scanning all keys is O(|memo|).
        for M, w_m in memo.items():
            rem = mask ^ M
            if rem in memo:
                score = 1
            else:
                # Approximate cost to build rem from scratch + 1 gate to XOR
                # build_from_bits uses popcount-1 gates. Total = popcount
                score = rem.bit_count()
            
            if score < min_score:
                min_score = score
                best_M = M
                best_rem = rem
                if score == 1:
                    # Cannot get better than 1 gate (0 is impossible as mask != M)
                    break

        # Check if best_rem is in memo
        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            # If best_rem is not in memo, build a tree from its bits
            bits = [j for j in range(n) if (best_rem >> j) & 1]
            w_rem = build_from_bits(bits)

        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # Sort rows by popcount descending to build strong bases early
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)

    outputs = [None] * m

    for idx in sorted_indices:
        if elapsed() > DEADLINE - 10.0:
            # Fast path: direct tree construction without search
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            outputs[idx] = compute_mask(row_masks[idx])

    # Fill any None (should be none)
    for i in range(m):
        if outputs[i] is None:
            # Fallback to zero (invalid if row != 0, but prevents crash)
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
