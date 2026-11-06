# EVOLVE-BLOCK-START
import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 45.0  # safety margin before 60s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to row masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Column weights (number of 1s in each column)
    col_weights = [0] * n
    for mask in row_masks:
        mm = mask
        while mm:
            b = mm & (-mm)
            col_weights[b.bit_length() - 1] += 1
            mm ^= b

    def elapsed():
        return time.time() - start_time

    def compute_basis_and_reps(col_order):
        """
        Given a column ordering, perform Gaussian elimination to find basis.
        Returns (basis_dict, row_reps) where:
          basis_dict: pivot_bit -> basis_mask
          row_reps: list of lists of basis indices for each row
        """
        basis_dict = {}  # pivot -> mask
        basis_list = []  # list of masks in order added
        basis_mask_to_idx = {}
        
        # Process rows
        row_reps = []
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_reps.append([])
                continue
            
            current_mask = mask
            rep = []
            
            # Reduce using existing basis
            # We need to reduce by trying each basis vector
            # To be efficient, we iterate through basis in order
            changed = True
            while changed:
                changed = False
                for b_idx, b_mask in enumerate(basis_list):
                    if (current_mask & b_mask) != 0 and (current_mask ^ b_mask).bit_count() < current_mask.bit_count():
                        # XORing reduces weight - do it
                        current_mask ^= b_mask
                        rep.append(b_idx)
                        changed = True
                        # Re-sort rep? No, just track
                # Alternative: standard elimination by pivot
            
            # Standard elimination: for each bit in current_mask from high to low
            cm = current_mask
            while cm:
                hb = cm.bit_length() - 1
                if hb in basis_dict:
                    b_mask = basis_dict[hb]
                    idx = basis_mask_to_idx[b_mask]
                    rep.append(idx)
                    cm ^= b_mask
                else:
                    break
            
            # If cm is still non-zero, this is a new basis vector
            if cm:
                hb = cm.bit_length() - 1
                b_idx = len(basis_list)
                basis_dict[hb] = cm
                basis_list.append(cm)
                basis_mask_to_idx[cm] = b_idx
                rep.append(b_idx)
            
            row_reps.append(rep)
        
        return basis_list, row_reps

    def estimate_cost(basis_list, row_reps):
        """
        Estimate total gate count without actually building gates.
        Uses a greedy memo-based estimate.
        """
        # Count unique masks that need to be built
        # Each basis vector needs gates to build from inputs
        # Each row combination needs gates to XOR basis wires together
        
        # For basis vectors: cost = popcount - 1 (XOR tree)
        basis_cost = sum(max(0, b.bit_count() - 1) for b in basis_list)
        
        # For row combinations: cost = len(rep) - 1 if len(rep) > 1
        # But with memoization, shared sub-expressions reduce cost
        # Simple estimate: sum of (len(rep) - 1) for all rows with len(rep) > 1
        row_cost = 0
        for rep in row_reps:
            if len(rep) > 1:
                row_cost += len(rep) - 1
            elif len(rep) == 1:
                row_cost += 0
        
        return basis_cost + row_cost

    def actual_synthesize(basis_list, row_reps):
        """
        Actually build the gates. Returns (gates, outputs, next_wire).
        """
        next_wire = n + 1
        gates = []
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

        def xor_tree(wires_masks):
            """
            wires_masks: list of (wire, mask) pairs
            Returns wire representing XOR of all masks.
            """
            if not wires_masks:
                return n
            if len(wires_masks) == 1:
                return wires_masks[0][0]
            
            wires = [w for w, m in wires_masks]
            masks = [m for w, m in wires_masks]
            
            while len(wires) > 1:
                new_wires = []
                new_masks = []
                for i in range(0, len(wires), 2):
                    if i + 1 < len(wires):
                        a_w, a_m = wires[i], masks[i]
                        b_w, b_m = wires[i+1], masks[i+1]
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

        # Build basis wires
        basis_wires = [None] * len(basis_list)
        for idx, b_mask in enumerate(basis_list):
            if b_mask in memo:
                basis_wires[idx] = memo[b_mask]
            else:
                bits = [j for j in range(n) if (b_mask >> j) & 1]
                if not bits:
                    basis_wires[idx] = n
                elif len(bits) == 1:
                    basis_wires[idx] = bits[0]
                else:
                    # Build XOR tree
                    wire_mask_pairs = [(j, 1 << j) for j in bits]
                    w = xor_tree(wire_mask_pairs)
                    memo[b_mask] = w
                    basis_wires[idx] = w

        # Build outputs
        outputs = [None] * m
        for i in range(m):
            rep = row_reps[i]
            if not rep:
                outputs[i] = n
            elif len(rep) == 1:
                outputs[i] = basis_wires[rep[0]]
            else:
                target_mask = row_masks[i]
                if target_mask in memo:
                    outputs[i] = memo[target_mask]
                else:
                    pairs = [(basis_wires[idx], basis_list[idx]) for idx in rep]
                    w = xor_tree(pairs)
                    memo[target_mask] = w
                    outputs[i] = w

        return gates, outputs, next_wire

    # Try multiple column orderings to find best basis
    # Strategy 1: sorted by column weight (ascending)
    # Strategy 2: sorted by column weight (descending)  
    # Strategy 3: natural order
    # Strategy 4: random permutations (limited by time)
    
    candidate_orders = []
    
    # Natural order
    candidate_orders.append(list(range(n)))
    
    # Sorted by column weight ascending
    candidate_orders.append(sorted(range(n), key=lambda j: col_weights[j]))
    
    # Sorted by column weight descending
    candidate_orders.append(sorted(range(n), key=lambda j: -col_weights[j]))
    
    # Try a few random permutations if time allows
    rng = random.Random(42)
    num_random = min(20, max(0, int((DEADLINE - elapsed()) * 10)))
    for _ in range(num_random):
        perm = list(range(n))
        rng.shuffle(perm)
        candidate_orders.append(perm)
    
    best_cost = float('inf')
    best_basis = None
    best_reps = None
    best_order = None
    
    for order in candidate_orders:
        if elapsed() > DEADLINE:
            break
        
        # Perform elimination with this column order
        # We need to modify the elimination to respect column order
        # Actually, the column order affects which columns are considered first
        # Let's implement elimination that processes columns in the given order
        
        basis_dict = {}
        basis_list = []
        basis_mask_to_idx = {}
        row_reps = []
        
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_reps.append([])
                continue
            
            current_mask = mask
            rep = []
            
            # Reduce by existing basis vectors
            # Standard: iterate through basis, XOR if it reduces
            # Better: use pivot-based elimination
            cm = current_mask
            while cm:
                hb = cm.bit_length() - 1
                if hb in basis_dict:
                    b_mask = basis_dict[hb]
                    idx = basis_mask_to_idx[b_mask]
                    rep.append(idx)
                    cm ^= b_mask
                else:
                    break
            
            if cm:
                hb = cm.bit_length() - 1
                b_idx = len(basis_list)
                basis_dict[hb] = cm
                basis_list.append(cm)
                basis_mask_to_idx[cm] = b_idx
                rep.append(b_idx)
            
            row_reps.append(rep)
        
        if not basis_list:
            # All zero rows
            cost = 0
        else:
            cost = estimate_cost(basis_list, row_reps)
        
        if cost < best_cost:
            best_cost = cost
            best_basis = basis_list
            best_reps = row_reps
            best_order = order
    
    # If no basis found (all zeros)
    if best_basis is None or len(best_basis) == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    # Actually synthesize with the best basis
    gates, outputs, _ = actual_synthesize(best_basis, best_reps)
    
    # Verify all outputs are set
    for i in range(m):
        if outputs[i] is None:
            outputs[i] = n
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
