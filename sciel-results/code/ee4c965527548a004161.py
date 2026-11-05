"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

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
    
    def smooth_benefit(count, is_cached, new_set, a, b):
        """
        Smooth surrogate scoring for pair selection.
        Accounts for:
        1. Immediate savings (count)
        2. Future sharing potential based on wire frequency
        """
        if is_cached:
            # No new gate, just replacement
            base = float(count)
        else:
            # New gate cost 1, saves (count - 1)
            base = float(count - 1)
        
        # Future bonus: wires that appear in many rows have higher future value
        # This is a smooth proxy for potential additional sharing
        freq_a = 0
        freq_b = 0
        for i in range(n_rows):
            if a in row_reprs[i]:
                freq_a += 1
            if b in row_reprs[i]:
                freq_b += 1
        
        # Smooth bonus based on frequency (log-scale to avoid domination)
        import math
        future_bonus = 0.05 * (math.log1p(freq_a) + math.log1p(freq_b))
        
        return base + future_bonus
    
    max_iter = 100000
    iter_count = 0
    
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
        
        # Find the best pair: maximize smooth benefit
        best_pair = None
        best_benefit = -1.0
        best_new_set = None
        best_cached_wire = None
        
        for (a, b), rows in pair_rows.items():
            # Compute the set that a XOR b represents
            set_a = wire_to_set.get(a, frozenset())
            set_b = wire_to_set.get(b, frozenset())
            new_set = set_a ^ set_b
            
            # Check if this set is already cached
            cached_wire = set_to_wire.get(new_set)
            
            # Count rows where this pair appears and row size >= 2
            count = 0
            for i in rows:
                if len(row_reprs[i]) >= 2:
                    count += 1
            
            if count < 2:
                continue
            
            is_cached = cached_wire is not None
            benefit = smooth_benefit(count, is_cached, new_set, a, b)
            
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
    
    # Final polish pass: try to find more sharing opportunities
    polish_deadline = start_time + 58.0
    for polish_iter in range(50):
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
