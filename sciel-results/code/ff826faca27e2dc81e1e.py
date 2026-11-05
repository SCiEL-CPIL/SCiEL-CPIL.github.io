"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
from itertools import combinations

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
    
    # Represent each row as a set of wire indices (initially input indices)
    row_reprs = [set(s) for s in row_sets]
    
    # Map from frozenset of input indices -> wire that computes it
    # Initially: each input wire i computes {i}, zero wire computes {}
    set_to_wire = {}
    for i in range(n_inputs):
        set_to_wire[frozenset([i])] = i
    set_to_wire[frozenset()] = zero_wire
    
    # Map from wire -> frozenset of input indices it represents
    wire_to_set = {}
    for i in range(n_inputs):
        wire_to_set[i] = frozenset([i])
    wire_to_set[zero_wire] = frozenset()
    
    gates = []
    next_wire = zero_wire + 1
    
    def compute_total_gates():
        total = len(gates)
        for rs in row_reprs:
            if len(rs) > 1:
                total += len(rs) - 1
        return total
    
    max_iter = 50000
    iter_count = 0
    
    def compute_cascading_benefit(a, b, rows):
        """Estimate cascading benefit: after merging a,b into new_wire,
        how many additional pairs in those rows become shared?"""
        set_a = wire_to_set.get(a, frozenset())
        set_b = wire_to_set.get(b, frozenset())
        new_set = set_a ^ set_b
        cached_wire = set_to_wire.get(new_set)
        
        # Count immediate savings
        count = 0
        for i in rows:
            if len(row_reprs[i]) >= 2:
                count += 1
        
        if count < 2:
            return 0, new_set, cached_wire
        
        # Estimate cascading: for each row containing (a,b), after merge,
        # check if new_wire forms new shared pairs with other wires in those rows
        cascade_bonus = 0
        new_wire_pairs = defaultdict(int)
        for i in rows:
            if len(row_reprs[i]) >= 2:
                rs = row_reprs[i]
                if a in rs and b in rs:
                    # After merge, new_wire will be in this row
                    # Count other wires in this row that also appear in other rows with new_wire
                    for w in rs:
                        if w != a and w != b:
                            new_wire_pairs[(w, i)].append(i) if False else None
                            # Just count occurrences of (w, new_wire_candidate) across rows
                            new_wire_pairs[w] = new_wire_pairs.get(w, 0) + 1
        
        # Wires that appear with new_wire in >= 2 rows give cascading benefit
        for w, cnt in new_wire_pairs.items():
            if cnt >= 2:
                cascade_bonus += cnt - 1
        
        if cached_wire is not None:
            benefit = count + cascade_bonus
        else:
            benefit = (count - 1) + cascade_bonus
        
        return benefit, new_set, cached_wire
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find all pairs of wires that appear together in at least 2 rows
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
        
        # Find the best pair: maximize benefit with cascading estimate
        best_pair = None
        best_benefit = 0
        best_new_set = None
        best_cached_wire = None
        
        for (a, b), rows in pair_rows.items():
            benefit, new_set, cached_wire = compute_cascading_benefit(a, b, rows)
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = (a, b)
                best_new_set = new_set
                best_cached_wire = cached_wire
        
        if best_pair is None:
            break
        
        a, b = best_pair
        
        if best_cached_wire is not None:
            # Use existing wire, no new gate
            new_wire = best_cached_wire
        else:
            # Create new gate
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_to_set[new_wire] = best_new_set
            set_to_wire[best_new_set] = new_wire
        
        # Update all rows that contain both a and b
        for i, rs in enumerate(row_reprs):
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                rs.add(new_wire)
    
    # Second phase: targeted sharing for large rows
    # For rows with many wires, try to find sub-expressions that can be shared
    max_iter2 = 20000
    iter_count2 = 0
    
    while iter_count2 < max_iter2:
        if time.time() > deadline:
            break
        iter_count2 += 1
        
        # Find rows with the most wires
        max_size = 0
        max_row = -1
        for i, rs in enumerate(row_reprs):
            if len(rs) > max_size:
                max_size = len(rs)
                max_row = i
        
        if max_size <= 2:
            break
        
        # For the largest row, try to extract sub-pairs that are shared
        rs = row_reprs[max_row]
        elems = sorted(rs)
        
        # Try all pairs in this row, prefer pairs whose XOR set appears in other rows too
        best_pair2 = None
        best_score2 = 0
        
        for x in range(len(elems)):
            for y in range(x + 1, len(elems)):
                a, b = elems[x], elems[y]
                
                # Check if (a,b) appears in other rows
                other_count = 0
                for j, rs2 in enumerate(row_reprs):
                    if j != max_row and a in rs2 and b in rs2:
                        other_count += 1
                
                if other_count < 1:
                    continue
                
                set_a = wire_to_set.get(a, frozenset())
                set_b = wire_to_set.get(b, frozenset())
                new_set = set_a ^ set_b
                cached_wire = set_to_wire.get(new_set)
                
                # Score: prefer cached wires and pairs shared across more rows
                score = other_count
                if cached_wire is not None:
                    score += 10  # Big bonus for no new gate
                
                if score > best_score2:
                    best_score2 = score
                    best_pair2 = (a, b)
                    best_new_set2 = new_set
                    best_cached_wire2 = cached_wire
        
        if best_pair2 is None:
            break
        
        a, b = best_pair2
        
        if best_cached_wire2 is not None:
            new_wire = best_cached_wire2
        else:
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_to_set[new_wire] = best_new_set2
            set_to_wire[best_new_set2] = new_wire
        
        # Update all rows
        for i, rs in enumerate(row_reprs):
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                rs.add(new_wire)
    
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
