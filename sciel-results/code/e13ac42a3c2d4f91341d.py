# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0  # Safety margin for 60s limit

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    def elapsed():
        return time.time() - start_time

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []

    # memo: bitmask -> wire_id
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_from_bits(bits_list):
        """Build balanced XOR tree from input bit indices, memoizing sub-masks."""
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]

        wires = list(bits_list)
        masks = [1 << b for b in bits_list]

        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w, a_m = wires[i], masks[i]
                    b_w, b_m = wires[i + 1], masks[i + 1]
                    new_m = a_m ^ b_m
                    if new_m in memo:
                        new_wires.append(memo[new_m])
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
        """Compute wire for mask using greedy nearest-neighbor memoization."""
        if mask in memo:
            return memo[mask]

        if elapsed() > DEADLINE - 10.0:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best existing mask M to minimize popcount(mask ^ M)
        best_M = 0
        best_rem = mask
        min_pop = mask.bit_count()

        for M, w_m in memo.items():
            rem = mask ^ M
            pc = rem.bit_count()
            if pc < min_pop:
                min_pop = pc
                best_M = M
                best_rem = rem
                if pc == 0:
                    break
                if pc == 1:
                    break

        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            bits = [j for j in range(n) if (best_rem >> j) & 1]
            w_rem = build_from_bits(bits)
            memo[best_rem] = w_rem

        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # Step 1: Gaussian elimination to find RREF basis
    basis = []  # list of (pivot_col, mask)
    working = list(row_masks)

    for col in range(n - 1, -1, -1):
        if elapsed() > DEADLINE - 10.0:
            break
        found = -1
        for i, mask in enumerate(working):
            if (mask >> col) & 1:
                found = i
                break
        if found == -1:
            continue
        pivot_mask = working[found]
        for i, mask in enumerate(working):
            if i != found and ((mask >> col) & 1):
                working[i] = mask ^ pivot_mask
        basis.append((col, pivot_mask))
        working[found] = 0

    # Step 2: Pre-compute wires for basis vectors using compute_mask
    for col, bmask in basis:
        if bmask not in memo:
            compute_mask(bmask)

    # Step 3: Process each row using compute_mask (greedy memoization)
    # Sort by popcount descending to build strong bases early
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)
    outputs = [None] * m

    for idx in sorted_indices:
        if elapsed() > DEADLINE - 5.0:
            # Fast fallback
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

    for i in range(m):
        if outputs[i] is None:
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
