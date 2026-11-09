def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    import random

    start_time = time.time()
    DEADLINE = 88.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integer bitmasks for fast XOR operations
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # Unique targets
    unique_targets = list(set(target_masks))
    
    # Initial available wires:
    # 0..n-1: inputs (bitmasks 1<<0, 1<<1, ...)
    # n: constant zero (0)
    
    # Strategy: 
    # The current program uses a greedy "global utility" heuristic which is slow and suboptimal.
    # A different structural approach is to use **Gaussian Elimination** to find a basis for the span of targets,
    # and then use **Common Subexpression Elimination (CSE)** via a DAG construction or a greedy 
    # "most frequent subexpression" heuristic to minimize gate count.
    
    # However, an even more powerful and structurally different approach for XOR circuits is 
    # **Basis-based Synthesis with Shared Intermediate Nodes**.
    
    # 1. Find a basis for the set of target masks.
    # 2. Express each target as a linear combination of the basis vectors.
    # 3. Instead of building each target independently, we build the basis vectors first.
    # 4. Then, for each target, we XOR the required basis vectors.
    # 5. To optimize further, we can look for common subexpressions in the target representations.
    
    # Let's implement a hybrid:
    # 1. Gaussian Elimination to get a basis.
    # 2. Use the basis to define a "virtual" coordinate system.
    # 3. Build the basis vectors using a greedy algorithm that minimizes depth and gates.
    # 4. Build targets by XORing basis vectors.
    # 5. Apply CSE: if a subset of basis vectors is XORed together for multiple targets, create a wire for that subset.
    
    # Step 1: Gaussian Elimination to find a basis for the target space.
    # We treat the targets as vectors in GF(2)^n.
    
    basis = []
    # Standard Gaussian elimination to find a basis for the span of target_masks
    # We want the basis vectors to be "simple" (low weight) if possible, but standard GE is fine.
    # Let's use a variant that tries to keep vectors sparse.
    
    # Actually, a simpler and often effective strategy for XOR circuits is:
    # **Greedy CSE on the set of targets.**
    # 1. Start with targets as "unsatisfied" sets of input wires.
    # 2. Repeatedly find the most common pair of wires (or subset) that can be factored out.
    # This is complex.
    
    # Let's try the **Basis + Balanced Tree** approach first, which is robust.
    # 1. Find a basis for the target masks.
    # 2. Build each basis vector using a balanced XOR tree of its constituent inputs.
    # 3. For each target, express it as a subset of basis vectors.
    # 4. Build each target using a balanced XOR tree of the required basis wires.
    # 5. This is independent synthesis. To improve, we can share the basis wires (which we do).
    # 6. To improve further, we can apply CSE on the target-side XORs.
    
    # Let's implement **Basis Synthesis with Target-Side CSE**.
    
    # 1. Find basis.
    basis_vectors = []
    # Use Gaussian Elimination to find a basis for the set of target masks.
    # We will store the basis as a list of masks.
    # We also need to know which inputs are in each basis vector.
    
    # Standard GE on the set of target masks.
    # We want to find a set of linearly independent vectors that span the targets.
    # Let's just use the targets themselves and reduce them.
    
    reduced_targets = []
    for t in sorted(set(target_masks), key=lambda x: x.bit_count()):
        # Try to reduce t using existing basis vectors
        current = t
        for bv in reduced_targets:
            if current & bv:
                # Find the highest bit set in both? Or just XOR if it reduces weight?
                # Standard GE: pivot on highest bit.
                # Let's use a simple greedy reduction: XOR with bv if it reduces popcount?
                # No, that's not guaranteed to find a basis.
                # Let's do proper GE.
                pass
        
    # Proper Gaussian Elimination to find a basis.
    # We want a basis for the subspace spanned by target_masks.
    # Let's use numpy for speed if n is large, but n <= 16 so pure Python is fine.
    
    # We'll represent vectors as integers.
    # Find a basis for the span of target_masks.
    basis = []
    # Sort by popcount to encourage sparsity?
    for t in sorted(set(target_masks), key=lambda x: x.bit_count()):
        v = t
        for b in basis:
            # If v and b share the highest set bit, XOR them
            # Find highest set bit in v
            if v == 0:
                break
            hb_v = v.bit_length() - 1
            hb_b = b.bit_length() - 1
            if hb_v == hb_b:
                v ^= b
        if v != 0:
            basis.append(v)
    
    # Now we have a basis.
    # We need to express each target as a combination of the basis vectors.
    # This is just Gaussian elimination again, but now we want the coefficients.
    
    # Let's build the basis vectors first.
    # For each basis vector, we build it from inputs.
    # To minimize gates, we can use a balanced tree or a greedy chain.
    # Balanced tree has depth log2(k) and k-1 gates.
    # Chain has depth k-1 and k-1 gates.
    # Since depth limit is 16, and n is at most 16, chain depth is at most 15.
    # So chain is safe for depth.
    # However, balanced tree might allow more CSE? No, CSE is on the logical structure.
    
    # Let's build each basis vector using a **Greedy Minimal XOR Tree**.
    # Actually, for a single vector, the minimal number of gates is popcount(v) - 1.
    # Any tree will have popcount(v) - 1 gates.
    # So we just need to pick a structure.
    
    # Let's use a **Left-Deep Tree** (Chain) for simplicity, unless we want to optimize depth.
    # Depth is not the primary metric, gate count is.
    # But depth limit is 16.
    # Max popcount is 16. Chain depth 15. Safe.
    
    # So, we build each basis vector with a chain.
    # wire_to_mask: dict mapping mask to wire index.
    # We start with inputs.
    
    wire_to_mask = {}
    mask_to_wire = {}
    gates = []
    next_wire = n + 1
    
    for i in range(n):
        mask = 1 << i
        wire_to_mask[i] = mask
        mask_to_wire[mask] = i
    
    wire_to_mask[n] = 0
    mask_to_wire[0] = n
    
    def get_wire_for_mask(mask):
        """Return wire index for mask, or create a new one if not available."""
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        # Create new wire for mask
        # We need to find two existing wires a, b such that a ^ b = mask.
        # This is the hard part. In this basis approach, we build basis vectors from inputs.
        # So we should build them explicitly.
        
        # This function is a fallback. For basis vectors, we will call a builder.
        # For now, return -1 to indicate failure.
        return -1

    # Function to build a mask from inputs, reusing existing wires if possible.
    def build_mask(mask):
        """Build a wire for the given mask, reusing existing wires."""
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        # Find a subset of available wires that XOR to mask.
        # Greedy: Find a wire w in available such that (mask ^ w) has smaller popcount?
        # Or just use inputs.
        
        # Let's use a recursive strategy:
        # 1. If mask is 0, return wire n.
        # 2. If mask is an input, return that wire.
        # 3. Find a wire w in available such that (mask ^ w) is "simpler".
        #    "Simpler" could mean lower popcount, or already in mask_to_wire.
        
        # For the basis vectors, we know they are built from inputs.
        # Let's just build them from inputs using a chain.
        
        bits = [i for i in range(n) if (mask >> i) & 1]
        if not bits:
            return n
        
        # Check if we can split mask into two parts that are already built or simpler.
        # This is essentially CSE.
        
        # Let's try to find a split mask = a ^ b where a and b are in mask_to_wire.
        # Iterate over all available masks?
        # For n=16, available masks might be small initially.
        
        # For now, let's just build from inputs if no split found.
        # But we want to reuse basis vectors for targets.
        
        # So, for targets, we will express them as XOR of basis vectors.
        # The basis vectors are built first.
        
        # So this function is only called for basis vectors and potentially intermediate CSE nodes.
        
        # Let's implement a simple CSE check:
        # Look for a pair (a, b) in mask_to_wire such that a ^ b == mask.
        # If found, create gate [wire_a, wire_b].
        
        # To make this efficient, we can iterate over the smaller set of masks.
        # But mask_to_wire grows.
        
        # Let's just build from inputs for the basis vectors, as they are the "atoms".
        
        curr_wire = bits[0]
        for b in bits[1:]:
            # Create gate for curr_wire ^ b
            # We need to ensure we don't create duplicate gates.
            # The resulting mask is mask_so_far ^ (1<<b).
            
            # Let's track the mask of curr_wire.
            # We don't have it directly, but we know it's the XOR of bits processed so far.
            
            # Let's refactor: build_mask should be smarter.
            
            pass
        
        # Let's restart the building strategy.
        # 1. Build basis vectors from inputs.
        # 2. Express targets in terms of basis.
        # 3. Build targets from basis wires, using CSE.
        
        return -1 # Placeholder

    # Let's implement a cleaner version.
    
    # 1. Build basis vectors.
    # For each basis vector, build it from inputs.
    # We will use a greedy algorithm to build each vector, reusing any existing wires.
    
    def build_vector_from_inputs(mask):
        """Build a wire for mask, assuming only inputs and zero are available initially."""
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        # Find a split: mask = a ^ b where a, b are in mask_to_wire.
        # If no split, build from inputs.
        
        # Try to find a split first.
        # Iterate over all masks in mask_to_wire.
        # This is O(N^2) in the number of wires.
        
        # For basis vectors, mask_to_wire only has inputs and zero.
        # So we can just build from inputs.
        
        bits = [i for i in range(n) if (mask >> i) & 1]
        if not bits:
            return n
        
        # Build a chain.
        curr_wire = bits[0]
        curr_mask = 1 << bits[0]
        
        for b in bits[1:]:
            # New mask is curr_mask ^ (1 << b)
            new_mask = curr_mask ^ (1 << b)
            
            # Check if new_mask is already in mask_to_wire.
            if new_mask in mask_to_wire:
                curr_wire = mask_to_wire[new_mask]
            else:
                # Create gate
                gates.append([curr_wire, b])
                new_wire = next_wire
                next_wire += 1
                mask_to_wire[new_mask] = new_wire
                wire_to_mask[new_wire] = new_mask
                curr_wire = new_wire
                curr_mask = new_mask
            
            # Update time check?
        
        return curr_wire

    # Build all basis vectors.
    basis_wires = []
    for bv in basis:
        w = build_vector_from_inputs(bv)
        basis_wires.append(w)
    
    # 2. Express each target in terms of the basis.
    # We need to find coefficients c_i such that target = XOR(c_i * basis_i).
    # This is solving a linear system.
    
    # Let's build a matrix B where rows are basis vectors.
    # We want to solve B * x = t for each target t.
    
    # Since basis is small (<= 16), we can use Gaussian elimination on the fly.
    
    def express_in_basis(target_mask):
        """Return a list of indices of basis vectors that XOR to target_mask."""
        # We have basis vectors in `basis`.
        # We want to find subset S such that XOR(basis[i] for i in S) == target_mask.
        
        # Greedy:
        current = target_mask
        used = []
        
        # Sort basis by popcount? Or just iterate.
        # Since basis is a basis, any vector can be uniquely expressed.
        # We can use the same GE logic.
        
        # Let's just iterate through basis vectors and XOR if it helps.
        # "Helps" means reduces the highest bit?
        
        # Standard GE:
        # We have a basis in row-echelon form (from our construction).
        # Our construction:
        # for t in sorted...:
        #   v = t
        #   for b in basis:
        #     if hb_v == hb_b: v ^= b
        #   if v != 0: basis.append(v)
        
        # This produces a basis where each vector has a unique highest bit.
        # So we can express any vector by looking at its highest bit, finding the basis vector
        # with that highest bit, and XORing it.
        
        # Let's map highest bit to basis vector index.
        hb_to_basis_idx = {}
        for i, bv in enumerate(basis):
            hb = bv.bit_length() - 1
            hb_to_basis_idx[hb] = i
        
        current = target_mask
        used = set()
        
        while current != 0:
            hb = current.bit_length() - 1
            if hb in hb_to_basis_idx:
                idx = hb_to_basis_idx[hb]
                current ^= basis[idx]
                used.add(idx)
            else:
                # Should not happen if basis spans the targets.
                break
        
        return sorted(used)

    # 3. Build targets from basis wires.
    # For each target, get the list of basis indices.
    # XOR the corresponding basis wires.
    # Use CSE: if a subset of basis wires is XORed together for multiple targets, share it.
    
    # Let's collect all target expressions.
    target_expressions = []
    for t in target_masks:
        expr = express_in_basis(t)
        target_expressions.append(expr)
    
    # Now, we have a set of sets (expressions).
    # We want to build a circuit that computes all these XORs.
    # This is a Multi-Output XOR Circuit.
    # We can use a greedy CSE:
    # 1. Count how many times each basis wire is used.
    # 2. Count how many times each pair of basis wires is used together?
    # 3. More generally, find common subexpressions.
    
    # A simple heuristic:
    # For each target, build a balanced tree of its basis wires.
    # If a subtree (XOR of a subset of basis wires) appears in multiple targets, create a shared wire.
    
    # Let's implement a simple version:
    # 1. For each target, if it has 0 basis wires, output 0.
    # 2. If it has 1, output that basis wire.
    # 3. If it has >1, XOR them.
    
    # To share, we can look for common prefixes or subsets.
    # Given time constraints, let's just build each target independently from the basis wires,
    # but reuse the basis wires themselves.
    
    # This is already an improvement over building from inputs for each target.
    # The basis wires are shared.
    
    # Can we do better?
    # Yes, by sharing intermediate XORs of basis wires.
    
    # Let's implement a **Greedy CSE on the target side**.
    # We maintain a set of "available" wires, which now includes the basis wires.
    # For each target, we try to build it by XORing available wires.
    # If we need to XOR two available wires, and the result is not available, we create a new wire.
    # If the result is already available, we use it.
    
    # This is effectively what `build_vector_from_inputs` did, but now the "inputs" are the basis wires.
    
    # Let's generalize `build_vector_from_inputs` to `build_from_available(mask)`.
    
    def build_from_available(mask):
        """Build a wire for mask using available wires (inputs, zero, basis wires, and previously created wires)."""
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        # Try to find a split mask = a ^ b where a, b are in mask_to_wire.
        # Iterate over all masks in mask_to_wire.
        # To speed up, we can iterate over the smaller set.
        # But mask_to_wire is a dict.
        
        # Let's iterate over all keys in mask_to_wire.
        # If |mask_to_wire| is large, this is slow.
        # But for n=16, the number of unique masks is at most 2^16, but likely much smaller.
        
        # Let's try to find a split.
        for a_mask, a_wire in list(mask_to_wire.items()):
            b_mask = mask ^ a_mask
            if b_mask in mask_to_wire:
                b_wire = mask_to_wire[b_mask]
                # Create gate
                gates.append([a_wire, b_wire])
                new_wire = next_wire
                next_wire += 1
                mask_to_wire[mask] = new_wire
                wire_to_mask[new_wire] = mask
                return new_wire
        
        # If no split found, we can't build it from existing wires.
        # This should not happen if we build targets in a way that ensures splits exist.
        # For targets expressed in basis, the basis wires are available.
        # So we should be able to build them by XORing basis wires.
        
        # If we fail, fallback to building from inputs?
        # Or just return -1.
        
        return -1

    # Now build all targets.
    outputs = []
    for t in target_masks:
        # t is a mask.
        # We need to build a wire for t.
        # We can use build_from_available(t).
        
        # But build_from_available might fail if no split is found.
        # To ensure success, we can build the target by explicitly XORing the basis wires.
        
        expr = express_in_basis(t)
        if not expr:
            outputs.append(n) # 0
        elif len(expr) == 1:
            outputs.append(basis_wires[expr[0]])
        else:
            # XOR the basis wires.
            # We can do this sequentially.
            curr_wire = basis_wires[expr[0]]
            curr_mask = basis[expr[0]]
            
            for idx in expr[1:]:
                next_mask = curr_mask ^ basis[idx]
                # Try to build next_mask from available.
                # It might already be there.
                if next_mask in mask_to_wire:
                    curr_wire = mask_to_wire[next_mask]
                else:
                    # Create gate
                    # We need a wire for curr_mask and a wire for basis[idx].
                    # curr_wire is for curr_mask.
                    # basis_wires[idx] is for basis[idx].
                    gates.append([curr_wire, basis_wires[idx]])
                    new_wire = next_wire
                    next_wire += 1
                    mask_to_wire[next_mask] = new_wire
                    wire_to_mask[new_wire] = next_mask
                    curr_wire = new_wire
                    curr_mask = next_mask
            
            outputs.append(curr_wire)
    
    return {'gates': gates, 'outputs': outputs}
