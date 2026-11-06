"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 52.0  # safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Parse rows as frozensets of column indices
    rows = []
    for i in range(m):
        row_set = frozenset(j for j in range(n) if matrix[i][j] % 2 == 1)
        rows.append(row_set)
    
    # Wire assignment: inputs are 0..n-1, const-zero is n
    # New gates get wires n+1, n+2, ...
    next_wire = n + 1
    gates = []
    
    # Memo: frozenset -> wire_id
    memo = {}
    for j in range(n):
        memo[frozenset([j])] = j
    memo[frozenset()] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
    
    def compute_set(s):
        """Compute wire for frozenset s, using memo and adding gates as needed."""
        if s in memo:
            return memo[s]
        
        if len(s) == 0:
            memo[s] = n
            return n
        if len(s) == 1:
            j = next(iter(s))
            memo[s] = j
            return j
        
        # Strategy 1: Find W in memo such that s ^ W is also in memo
        for W, wire_w in memo.items():
            r = s ^ W
            if r in memo:
                wire_r = memo[r]
                new_w = add_gate(wire_w, wire_r)
                memo[s] = new_w
                return new_w
        
        # Strategy 2: Find W in memo that minimizes popcount(s ^ W)
        best_cost = float('inf')
        best_W = None
        best_r = None
        for W, wire_w in memo.items():
            r = s ^ W
            cost = len(r)
            if cost < best_cost:
                best_cost = cost
                best_W = W
                best_r = r
                if cost <= 2:
                    break
        
        if best_W is not None:
            wire_w = memo[best_W]
            if best_r in memo:
                wire_r = memo[best_r]
            else:
                wire_r = compute_set(best_r)
            new_w = add_gate(wire_w, wire_r)
            memo[s] = new_w
            return new_w
        
        # Fallback: balanced XOR tree
        elems = list(s)
        wires = [j for j in elems]
        while len(wires) > 1:
            new_wires = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    new_w = add_gate(wires[i], wires[i+1])
                    new_wires.append(new_w)
                else:
                    new_wires.append(wires[i])
            wires = new_wires
        result_wire = wires[0]
        memo[s] = result_wire
        return result_wire
    
    # === Improved approach: Gaussian elimination over GF(2) to find minimal basis ===
    # Convert rows to integer bitmasks for efficient GF(2) operations
    def row_to_mask(row_set):
        mask = 0
        for j in row_set:
            mask |= (1 << j)
        return mask
    
    row_masks = [row_to_mask(r) for r in rows]
    
    # Gaussian elimination to find row-reduced echelon form
    # This gives us a minimal set of independent rows
    # We'll use this to identify which rows are linear combinations of others
    
    # Build a list of (mask, index) and reduce
    reduced = []
    pivot_cols = {}  # col -> (mask, index)
    
    for idx, mask in enumerate(row_masks):
        cur = mask
        for col in range(n):
            if cur & (1 << col):
                if col in pivot_cols:
                    pmask, _ = pivot_cols[col]
                    cur ^= pmask
                else:
                    break
        if cur == 0:
            continue
        # Find pivot
        for col in range(n):
            if cur & (1 << col):
                pivot_cols[col] = (cur, idx)
                reduced.append((cur, idx))
                break
    
    # Now we have reduced rows. The original rows can be expressed as combinations.
    # For gate minimization, we want to share intermediate XORs.
    
    # New strategy: Use a more systematic approach
    # 1. Compute all row masks
    # 2. Find a good set of intermediate masks to pre-compute
    # 3. Build the circuit
    
    # Let's use a different approach: sort rows by popcount and greedily build
    # Pre-compute all possible 2-way XORs of rows, then use them to build higher orders
    
    # Reset state for the improved approach
    next_wire = n + 1
    gates = []
    memo = {}
    for j in range(n):
        memo[frozenset([j])] = j
    memo[frozenset()] = n
    
    # First pass: compute all rows using compute_set, but with better ordering
    # Sort rows by popcount to build up from simple to complex
    row_order = sorted(range(m), key=lambda i: (len(rows[i]), i))
    
    outputs = [None] * m
    
    # Pre-compute useful intermediates: pairwise XORs of similar-popcount rows
    if m <= 40 and elapsed() < DEADLINE - 20.0:
        # Group rows by approximate popcount
        popcounts = [(len(rows[i]), i) for i in range(m)]
        popcounts.sort()
        
        # For rows with similar popcount, pre-compute their XOR
        computed_pairs = set()
        for a_idx in range(len(popcounts)):
            if elapsed() > DEADLINE - 15.0:
                break
            pa, ia = popcounts[a_idx]
            for b_idx in range(a_idx + 1, min(a_idx + 10, len(popcounts))):
                if elapsed() > DEADLINE - 15.0:
                    break
                pb, ib = popcounts[b_idx]
                if abs(pa - pb) > 4:
                    continue
                xor_set = rows[ia] ^ rows[ib]
                key = (min(ia, ib), max(ia, ib))
                if key in computed_pairs:
                    continue
                computed_pairs.add(key)
                if 0 < len(xor_set) <= max(6, pa // 2):
                    # Pre-compute this XOR
                    compute_set(xor_set)
    
    # Now compute all rows in sorted order
    for idx in row_order:
        if elapsed() > DEADLINE - 5.0:
            break
        s = rows[idx]
        outputs[idx] = compute_set(s)
    
    # Fill in any missing outputs
    for i in range(m):
        if outputs[i] is None:
            if elapsed() > DEADLINE - 3.0:
                outputs[i] = n
            else:
                outputs[i] = compute_set(rows[i])
    
    while len(outputs) < m:
        outputs.append(n)
    
    # Second optimization pass: try to find shared subexpressions
    if elapsed() < DEADLINE - 10.0 and len(gates) > 5:
        # Try to identify patterns where the same intermediate is computed multiple times
        # This is hard to detect directly, so instead try a different construction
        
        # Alternative: use a priority queue approach
        # Start with all row masks, repeatedly merge the two with maximum overlap
        next_wire2 = n + 1
        gates2 = []
        memo2 = {}
        for j in range(n):
            memo2[frozenset([j])] = j
        memo2[frozenset()] = n
        
        def add_gate2(a, b):
            nonlocal next_wire2
            gates2.append([a, b])
            w = next_wire2
            next_wire2 += 1
            return w
        
        # Build a worklist of sets to compute
        # Start with all rows
        to_compute = set(rows)
        # Add their pairwise XORs if cheap
        row_list = list(to_compute)
        if len(row_list) <= 25:
            for i in range(len(row_list)):
                for j in range(i + 1, len(row_list)):
                    xor_s = row_list[i] ^ row_list[j]
                    if 0 < len(xor_s) <= 8:
                        to_compute.add(xor_s)
        
        # Sort by popcount
        to_compute = sorted(to_compute, key=lambda s: (len(s), hash(s)))
        
        # Compute in order, using memoization
        def compute_set2(s):
            if s in memo2:
                return memo2[s]
            if len(s) == 0:
                memo2[s] = n
                return n
            if len(s) == 1:
                j = next(iter(s))
                memo2[s] = j
                return j
            
            # Try to find decomposition
            for W in memo2:
                r = s ^ W
                if r in memo2:
                    new_w = add_gate2(memo2[W], memo2[r])
                    memo2[s] = new_w
                    return new_w
            
            # Find best single-element reduction
            best_cost = float('inf')
            best_W = None
            for W in memo2:
                r = s ^ W
                cost = len(r)
                if cost < best_cost:
                    best_cost = cost
                    best_W = W
                    if cost <= 1:
                        break
            
            if best_W is not None:
                r = s ^ best_W
                wire_r = compute_set2(r)
                new_w = add_gate2(memo2[best_W], wire_r)
                memo2[s] = new_w
                return new_w
            
            # Fallback: balanced tree
            elems = list(s)
            wires = list(elems)
            while len(wires) > 1:
                new_wires = []
                for i in range(0, len(wires), 2):
                    if i + 1 < len(wires):
                        new_w = add_gate2(wires[i], wires[i+1])
                        new_wires.append(new_w)
                    else:
                        new_wires.append(wires[i])
                wires = new_wires
            result = wires[0]
            memo2[s] = result
            return result
        
        # Pre-compute the sorted sets
        for cs in to_compute:
            if elapsed() > DEADLINE - 5.0:
                break
            compute_set2(cs)
        
        # Compute all rows
        outputs2 = [None] * m
        for i in range(m):
            if elapsed() > DEADLINE - 3.0:
                break
            outputs2[i] = compute_set2(rows[i])
        
        # Use the better result
        if all(o is not None for o in outputs2) and len(gates2) < len(gates):
            gates = gates2
            outputs = outputs2
    
    while len(outputs) < m:
        outputs.append(n)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
