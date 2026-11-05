"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import math

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Convert rows to sets of input indices
    row_sets = []
    for i in range(n_rows):
        s = frozenset(np.where(A[i] == 1)[0])
        row_sets.append(s)
    
    # Strategy: Greedy shared-subexpression with wire merging and cancellation
    # Enhanced with Minimum Weight Basis Completion insight:
    # Prioritize creating low-weight intermediates that maximize row sharing
    
    row_reprs = [set(s) for s in row_sets]  # mutable sets of wire indices
    wire_to_set = {i: frozenset([i]) for i in range(n_inputs)}
    wire_to_set[zero_wire] = frozenset()
    wire_depth = {i: 0 for i in range(n_inputs)}
    wire_depth[zero_wire] = 0
    gates = []
    next_wire = zero_wire + 1
    
    # Cache: frozenset -> wire index (for merging identical intermediates)
    set_to_wire = {frozenset([i]): i for i in range(n_inputs)}
    set_to_wire[frozenset()] = zero_wire
    
    def get_or_create_wire(wire_a, wire_b):
        """Get existing wire for XOR of wire_a and wire_b, or create a new one."""
        nonlocal next_wire
        new_set = wire_to_set[wire_a] ^ wire_to_set[wire_b]
        # Check if we already have a wire with this set
        if new_set in set_to_wire:
            return set_to_wire[new_set], False
        # Create new wire
        new_wire = next_wire
        next_wire += 1
        wire_to_set[new_wire] = new_set
        wire_depth[new_wire] = max(wire_depth[wire_a], wire_depth[wire_b]) + 1
        set_to_wire[new_set] = new_wire
        gates.append([wire_a, wire_b])
        return new_wire, True
    
    def compute_total_gates():
        total = len(gates)
        for rs in row_reprs:
            if len(rs) > 1:
                total += len(rs) - 1
        return total
    
    # Smooth surrogate scoring combining:
    # 1. Immediate gate savings
    # 2. Future sharing potential (low-weight intermediates)
    # 3. Cancellation bonus
    # 4. Depth penalty
    def compute_benefit(count, is_existing, new_set, a, b):
        new_set_size = len(new_set)
        
        # Base benefit: saves (count - 1) gates in final XOR tree
        benefit = float(count - 1)
        
        # If the set is empty (cancellation), each row loses 2 wires and gains 0
        # This saves an extra gate per row
        if new_set_size == 0:
            benefit += float(count)
        
        # If the set already exists, we don't create a new gate, so extra saving
        if is_existing:
            benefit += 3.0
        
        # Minimum Weight Basis Completion insight:
        # Small intermediates are more likely to be shared across many rows
        # Use a smooth decay: bonus ~ 1/sqrt(size) for small sizes
        if 0 < new_set_size <= 2:
            benefit += 2.5
        elif new_set_size == 3:
            benefit += 1.8
        elif new_set_size == 4:
            benefit += 1.2
        elif new_set_size <= 8:
            benefit += 0.5 / math.sqrt(new_set_size)
        
        # Bonus if a or b themselves are intermediate wires
        if a >= zero_wire + 1 or b >= zero_wire + 1:
            benefit += 0.5
        
        # Depth penalty: prefer shallower circuits
        new_depth = max(wire_depth[a], wire_depth[b]) + 1
        if new_depth > depth_limit:
            return -1e9  # Skip pairs that would exceed depth limit
        benefit -= new_depth * 0.1
        
        # Future sharing potential: check if new_set appears as subset in other rows
        # This is a proxy for future reduction opportunities
        if not is_existing and 0 < new_set_size <= 4:
            subset_count = 0
            for i in range(n_rows):
                rs_i = row_reprs[i]
                if new_set.issubset(rs_i) and len(rs_i) > new_set_size:
                    subset_count += 1
            benefit += 0.3 * subset_count
        
        return benefit
    
    max_iter = 50000
    iter_count = 0
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find all pairs that appear together in at least 2 rows
        pair_rows = defaultdict(list)
        for i, rs in enumerate(row_reprs):
            if len(rs) >= 2:
                elems = sorted(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        pair_rows[(a, b)].append(i)
        
        if not pair_rows:
            break
        
        # Find the best pair with enhanced benefit scoring
        best_pair = None
        best_benefit = 0
        best_new_set = None
        best_is_existing = False
        
        for (a, b), rows in pair_rows.items():
            # Count rows where size >= 2
            count = 0
            for i in rows:
                if len(row_reprs[i]) >= 2:
                    count += 1
            if count < 2:
                continue
            
            new_set = wire_to_set[a] ^ wire_to_set[b]
            is_existing = new_set in set_to_wire
            
            benefit = compute_benefit(count, is_existing, new_set, a, b)
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = (a, b)
                best_new_set = new_set
                best_is_existing = is_existing
        
        if best_pair is None:
            break
        
        a, b = best_pair
        
        # Get or create the wire for a XOR b
        new_wire, was_created = get_or_create_wire(a, b)
        
        # Update all rows that contain both a and b
        for i, rs in enumerate(row_reprs):
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                # Only add new_wire if it's not the zero wire (cancellation)
                if new_wire != zero_wire:
                    rs.add(new_wire)
    
    # Polish pass: try to find more sharing opportunities with tighter criteria
    polish_deadline = start_time + 58.0
    for polish_iter in range(100):
        if time.time() > polish_deadline:
            break
        
        # Look for pairs that appear in >= 2 rows
        pair_count = defaultdict(int)
        for i, rs in enumerate(row_reprs):
            if len(rs) >= 2:
                elems = sorted(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        pair_count[(a, b)] += 1
        
        # Try to merge the most common pair
        best_pair = None
        best_count = 1
        for (a, b), cnt in pair_count.items():
            if cnt > best_count:
                best_count = cnt
                best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        set_a = wire_to_set.get(a, frozenset())
        set_b = wire_to_set.get(b, frozenset())
        new_set = set_a ^ set_b
        
        cached_wire = set_to_wire.get(new_set)
        if cached_wire is not None:
            new_wire = cached_wire
        else:
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_to_set[new_wire] = new_set
            set_to_wire[new_set] = new_wire
        
        changed = False
        for i, rs in enumerate(row_reprs):
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                if new_wire != zero_wire:
                    rs.add(new_wire)
                changed = True
        
        if not changed:
            break
    
    # Build final outputs: for each row, XOR all its wires together
    outputs = []
    for i in range(n_rows):
        rs = row_reprs[i]
        if len(rs) == 0:
            outputs.append(zero_wire)
        elif len(rs) == 1:
            outputs.append(list(rs)[0])
        else:
            elems = sorted(rs)
            current = elems[0]
            for e in elems[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
