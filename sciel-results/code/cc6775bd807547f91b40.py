import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 85.0

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

    def get_basis_and_coords(row_masks, n):
        """
        Finds a basis for the row space and coordinates of each row in that basis.
        Returns (basis_masks, coords) where coords[i] is a bitmask of basis indices.
        """
        # 1. Find a basis.
        # We want a sparse basis.
        # Greedy: pick independent rows with lowest popcount.
        
        # First, filter out zeros
        non_zero_rows = [m for m in row_masks if m != 0]
        
        # Sort by popcount
        non_zero_rows.sort(key=lambda x: x.bit_count())
        
        basis = []
        # We need to check independence.
        # To check if a mask is in the span of basis, we can reduce it.
        # We maintain basis in a form that allows fast reduction?
        # For small n (<=16), we can just check if the rank increases.
        # Or simpler: keep a list of basis vectors.
        # To check if `mask` is in span of `basis`:
        #   Try to reduce `mask` by XORing with vectors in `basis`.
        #   If we can reduce it to 0, it's dependent.
        #   This is O(|basis|) if we are lucky, but worst case O(2^|basis|) if we do it naively.
        #   However, n is small. We can use Gaussian elimination on the fly.
        
        # Let's use a standard RREF approach to find a basis.
        # We'll maintain a list of basis vectors in RREF.
        # pivot_to_basis: dict {pivot_bit: mask}
        # When adding a new vector, we reduce it by existing pivots.
        # If it becomes 0, it's dependent.
        # Else, we find its pivot, add it to basis, and eliminate this pivot from all existing basis vectors.
        
        pivot_to_basis = {}
        basis_list = []
        
        for mask in non_zero_rows:
            temp = mask
            # Reduce temp by existing pivots
            # We need to iterate through pivots in a specific order?
            # Actually, if we maintain RREF, the pivots are distinct.
            # We can just XOR with any basis vector that has the same highest bit as temp?
            # No, RREF means if temp has bit p set, and there is a basis vector with pivot p,
            # we XOR them. This eliminates bit p from temp.
            # We repeat until temp is 0 or its highest bit is not a pivot.
            
            changed = True
            while changed:
                changed = False
                # Find highest set bit in temp
                if temp == 0:
                    break
                highest_bit = temp.bit_length() - 1
                if highest_bit in pivot_to_basis:
                    temp ^= pivot_to_basis[highest_bit]
                    changed = True
            
            if temp == 0:
                continue # Dependent
            
            # temp is independent
            # Find its pivot (highest set bit)
            pivot_bit = temp.bit_length() - 1
            
            # Add to basis
            # First, eliminate this pivot from all existing basis vectors to maintain RREF
            for p, b in list(pivot_to_basis.items()):
                if (b >> pivot_bit) & 1:
                    new_b = b ^ temp
                    pivot_to_basis[p] = new_b
            
            # Add new vector
            pivot_to_basis[pivot_bit] = temp
            basis_list.append(temp)
            
        # Now we have a basis in RREF.
        # 2. Find coordinates of each original row in this basis.
        # We reduce the original row using the basis vectors.
        # The XORs we perform give the coordinates.
        
        coords = []
        for mask in row_masks:
            temp = mask
            coord_mask = 0
            # We need to map basis vectors to indices.
            # basis_list[i] is the i-th basis vector.
            # pivot_to_basis maps pivot_bit -> mask.
            # We need to know which index in basis_list corresponds to the mask we XOR with.
            
            # Let's create a map: mask -> index
            mask_to_idx = {b: i for i, b in enumerate(basis_list)}
            
            changed = True
            while changed:
                changed = False
                if temp == 0:
                    break
                highest_bit = temp.bit_length() - 1
                if highest_bit in pivot_to_basis:
                    b_mask = pivot_to_basis[highest_bit]
                    b_idx = mask_to_idx[b_mask]
                    temp ^= b_mask
                    coord_mask ^= (1 << b_idx)
                    changed = True
            
            # If temp is not 0, something went wrong (should not happen if basis is correct)
            coords.append(coord_mask)
            
        return basis_list, coords

    basis_masks, coords = get_basis_and_coords(row_masks, n)
    
    # 3. Synthesize basis vectors
    # We use a recursive memoized builder.
    # Since basis vectors are independent, we build them one by one.
    # We can reuse sub-expressions from previous basis vectors if they are in memo.
    
    basis_wires = []
    
    def build_mask(mask):
        if mask in memo:
            return memo[mask]
        
        # Find best split
        # We want to minimize the number of new gates.
        # Ideally, mask = A ^ B where A, B are in memo.
        # If not, A in memo, B is simple.
        
        best_cost = float('inf')
        best_A = 0
        best_B = mask
        best_A_wire = None
        best_B_wire = None
        
        # Check all M in memo
        for M, w_m in memo.items():
            rem = mask ^ M
            if rem in memo:
                cost = 1 # 1 gate to combine
                if cost < best_cost:
                    best_cost = cost
                    best_A = M
                    best_B = rem
                    best_A_wire = w_m
                    best_B_wire = memo[rem]
                    break # Can't do better than 1 gate
            
            # If rem is not in memo, estimate cost
            # Cost is roughly popcount(rem) if we build it from scratch
            # But we can recursively build it.
            # To avoid deep recursion, we only consider this if rem is "small"
            # Or we just use popcount as a heuristic.
            # Let's use popcount as a proxy for cost of building rem.
            # This is a heuristic.
            
            # Actually, if we don't have a good estimate, we might pick a bad split.
            # Let's stick to splits where one part is in memo and the other is "simple" (popcount <= 2)
            # Or just use the first M that gives a small rem.
            
            rem_pop = rem.bit_count()
            if rem_pop < best_cost:
                best_cost = rem_pop
                best_A = M
                best_B = rem
                best_A_wire = w_m
                # best_B_wire will be built recursively or from scratch
        
        # If we found a split where both are in memo, we are done.
        if best_A_wire is not None and best_B_wire is not None:
            w_new = add_gate(best_A_wire, best_B_wire)
            memo[mask] = w_new
            return w_new
        
        # Otherwise, we have a best_A (in memo) and best_B (not in memo).
        # We need to build best_B.
        # To avoid infinite recursion, we check if best_B is "small" enough to build from scratch
        # or if we should just build it recursively.
        
        # If best_B is 0, it's in memo (0).
        # If best_B is a single bit, it's in memo.
        
        # Let's recursively build best_B.
        # But we must be careful not to loop.
        # Since we are adding `mask` to memo at the end, and `best_B` is a sub-part,
        # it's possible that `best_B` is related to `mask` in a way that causes a loop?
        # No, `best_B = mask ^ best_A`. If `best_A` is non-zero, `best_B` is different from `mask`.
        # If `best_A` is 0, `best_B = mask`, which would be a loop.
        # So we should avoid `best_A = 0` if `mask` is not in memo.
        # But `0` is in memo.
        
        # If `best_A` was 0, we should have found `mask` in memo if it was possible?
        # No, `mask` is not in memo.
        # So we should not pick `best_A = 0` unless `best_B` is simpler than `mask`.
        
        # Let's refine the search:
        # Only consider M if `mask ^ M` is "simpler" than `mask`.
        # "Simpler" means smaller popcount?
        
        # Let's restart the split search with this constraint.
        
        best_cost = float('inf')
        best_A = 0
        best_B = mask
        best_A_wire = None
        best_B_wire = None
        
        for M, w_m in memo.items():
            rem = mask ^ M
            if rem == mask: # M=0
                continue
            
            if rem in memo:
                cost = 1
                if cost < best_cost:
                    best_cost = cost
                    best_A = M
                    best_B = rem
                    best_A_wire = w_m
                    best_B_wire = memo[rem]
                    break
            
            rem_pop = rem.bit_count()
            if rem_pop < mask.bit_count() and rem_pop < best_cost:
                best_cost = rem_pop
                best_A = M
                best_B = rem
                best_A_wire = w_m
        
        if best_A_wire is not None and best_B_wire is not None:
            w_new = add_gate(best_A_wire, best_B_wire)
            memo[mask] = w_new
            return w_new
        
        if best_A_wire is not None:
            # Build best_B recursively
            w_b = build_mask(best_B)
            w_new = add_gate(best_A_wire, w_b)
            memo[mask] = w_new
            return w_new
        
        # Fallback: build from scratch using tree
        bits = [j for j in range(n) if (mask >> j) & 1]
        if not bits:
            w_new = n
        elif len(bits) == 1:
            w_new = bits[0]
        else:
            wires = bits[:]
            while len(wires) > 1:
                new_wires = []
                for i in range(0, len(wires), 2):
                    if i + 1 < len(wires):
                        new_w = add_gate(wires[i], wires[i+1])
                        new_wires.append(new_w)
                    else:
                        new_wires.append(wires[i])
                wires = new_wires
            w_new = wires[0]
            
        memo[mask] = w_new
        return w_new

    # Build basis vectors
    for b_mask in basis_masks:
        w = build_mask(b_mask)
        basis_wires.append(w)
        
    # 4. Synthesize outputs
    outputs = []
    for coord in coords:
        if coord == 0:
            outputs.append(n)
        else:
            # XOR the basis wires indicated by coord
            wires_to_xor = []
            for i in range(len(basis_wires)):
                if (coord >> i) & 1:
                    wires_to_xor.append(basis_wires[i])
            
            if not wires_to_xor:
                outputs.append(n)
            elif len(wires_to_xor) == 1:
                outputs.append(wires_to_xor[0])
            else:
                # Build a tree to XOR them
                wires = wires_to_xor[:]
                while len(wires) > 1:
                    new_wires = []
                    for i in range(0, len(wires), 2):
                        if i + 1 < len(wires):
                            new_w = add_gate(wires[i], wires[i+1])
                            new_wires.append(new_w)
                        else:
                            new_wires.append(wires[i])
                    wires = new_wires
                outputs.append(wires[0])
                
    return {'gates': gates, 'outputs': outputs}
