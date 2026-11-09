"""XOR circuit synthesis with CSE, adaptive ordering, and optimized decomposition."""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Remove duplicate rows (they share the same wire)
    row_to_base = {}
    unique_rows = []
    for i in range(m):
        r = rows[i]
        if r not in row_to_base:
            row_to_base[r] = len(unique_rows)
            unique_rows.append(r)
        else:
            pass  # already mapped

    # Build the circuit for unique rows
    val_to_wire = {0: n}  # constant zero
    for j in range(n):
        val_to_wire[1 << j] = j

    gates = []
    next_wire = n + 1

    def popcount(x):
        return bin(x).count('1')

    def find_one_gate_solution(target):
        """Check if target can be made with one XOR gate from existing wires."""
        for va, wa in val_to_wire.items():
            vb = target ^ va
            if vb in val_to_wire:
                wb = val_to_wire[vb]
                if wa != wb:
                    return (wa, wb)
        return None

    def estimate_cost(target):
        """Estimate gate count to build target from current wires."""
        if target == 0:
            return 0
        if target in val_to_wire:
            return 0
        # Check one-gate solution
        if find_one_gate_solution(target) is not None:
            return 1
        # Estimate: 1 + min over helpers of (cost to build diff)
        # Use popcount as a proxy for multi-gate cost
        min_est = float('inf')
        for existing_val in val_to_wire:
            diff = target ^ existing_val
            if diff == 0:
                min_est = 0
                break
            # diff can be built with at most popcount(diff) gates (XOR with individual inputs)
            est = 1 + popcount(diff)
            if est < min_est:
                min_est = est
        return min_est

    def build_from_inputs(target):
        """Build target from input wires using minimal gates."""
        nonlocal next_wire, gates
        if target == 0:
            return n
        if target in val_to_wire:
            return val_to_wire[target]

        # Check one-gate solution first
        one_gate = find_one_gate_solution(target)
        if one_gate is not None:
            w1, w2 = one_gate
            gates.append([w1, w2])
            new_wire = next_wire
            next_wire += 1
            val_to_wire[target] = new_wire
            return new_wire

        # Find sparsest decomposition
        best_diff = target
        best_helper = None
        for existing_val, existing_wire in val_to_wire.items():
            diff = target ^ existing_val
            if diff == 0:
                return existing_wire
            if best_helper is None or popcount(diff) < popcount(best_diff):
                best_diff = diff
                best_helper = existing_wire

        if best_diff == 0:
            return best_helper

        # Recursively build the difference
        diff_wire = build_from_inputs(best_diff)
        gates.append([best_helper, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wire[target] = new_wire
        return new_wire

    # Phase 2: Iterative Greedy Intermediate Addition
    # Dynamically generate candidates and pick the one with highest gain
    num_unique = len(unique_rows)
    max_iter = 50 # Limit iterations to avoid timeout
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate candidates from current state
        candidates = set()
        
        # 1. Row ^ Row
        for i in range(num_unique):
            for j in range(i + 1, num_unique):
                c = unique_rows[i] ^ unique_rows[j]
                if c != 0 and c not in val_to_wire:
                    candidates.add(c)
                    
        # 2. Row ^ Input
        for r in unique_rows:
            for j in range(n):
                c = r ^ (1 << j)
                if c != 0 and c not in val_to_wire:
                    candidates.add(c)
                    
        # 3. Intermediate ^ Intermediate (only if few exist to keep it fast)
        if len(val_to_wire) < 20:
            vals = list(val_to_wire.keys())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    c = vals[i] ^ vals[j]
                    if c != 0 and c not in val_to_wire:
                        candidates.add(c)

        if not candidates:
            break

        # Evaluate gain for each candidate
        best_cand = None
        best_gain = 0
        
        for c in candidates:
            gain = 0
            for r in unique_rows:
                if r in val_to_wire:
                    continue
                
                # Current estimated cost
                min_c = popcount(r)
                for w in val_to_wire:
                    diff = r ^ w
                    if diff == 0:
                        min_c = 0
                        break
                    c_est = 1 + popcount(diff)
                    if c_est < min_c:
                        min_c = c_est
                
                # New estimated cost with c
                diff_c = r ^ c
                if diff_c == 0:
                    new_c = 0
                else:
                    new_c = 1 + popcount(diff_c)
                    if new_c > min_c:
                        new_c = min_c
                
                gain += (min_c - new_c)
                
            if gain > best_gain:
                best_gain = gain
                best_cand = c

        if best_cand is None or best_gain <= 0:
            break
        
        # Build the best candidate
        build_from_inputs(best_cand)

    # Phase 3: Build remaining unique rows greedily by cost
    row_wire = {}  # unique_row_value -> wire
    still_remaining = []
    for i in range(num_unique):
        r = unique_rows[i]
        if r == 0:
            row_wire[r] = n
        elif r in val_to_wire:
            row_wire[r] = val_to_wire[r]
        else:
            still_remaining.append(i)

    while still_remaining:
        if time.time() > deadline:
            break

        # Find row with minimum estimated cost
        min_cost = float('inf')
        min_idx = -1
        for i in still_remaining:
            c = estimate_cost(unique_rows[i])
            if c < min_cost:
                min_cost = c
                min_idx = i

        if min_idx == -1:
            break

        target = unique_rows[min_idx]

        if target in val_to_wire:
            row_wire[target] = val_to_wire[target]
            still_remaining.remove(min_idx)
            continue

        wire = build_from_inputs(target)
        row_wire[target] = wire
        still_remaining.remove(min_idx)

    # Ensure all unique rows have wires
    for i in range(num_unique):
        r = unique_rows[i]
        if r not in row_wire:
            if r == 0:
                row_wire[r] = n
            elif r in val_to_wire:
                row_wire[r] = val_to_wire[r]
            else:
                row_wire[r] = build_from_inputs(r)

    # Map back to original row indices
    outputs = [row_wire[rows[i]] for i in range(m)]

    return {'gates': gates, 'outputs': outputs}
