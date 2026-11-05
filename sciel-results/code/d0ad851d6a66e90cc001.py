"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

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
    deadline = start_time + 58.0  # Safety margin before 60s limit

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Optimization: Identify unique rows to reduce search space
    unique_rows = {}
    row_to_unique = []
    unique_row_indices = []
    
    for i in range(n_rows):
        row_tuple = tuple(A[i].tolist())
        if row_tuple not in unique_rows:
            unique_rows[row_tuple] = len(unique_rows)
            unique_row_indices.append(row_tuple)
        row_to_unique.append(unique_rows[row_tuple])
        
    n_unique = len(unique_rows)
    
    # Initialize state for unique rows
    row_wires_unique = []
    for i in range(n_unique):
        row_data = unique_row_indices[i]
        if sum(row_data) == 0:
            row_wires_unique.append(set())
        else:
            row_wires_unique.append(set(np.where(np.array(row_data) == 1)[0].tolist()))
    
    gates = []
    next_wire = zero_wire + 1
    
    # Map from wire to its underlying set of input indices (for tracking)
    wire_set = {}
    for i in range(n_inputs):
        wire_set[i] = frozenset([i])
    wire_set[zero_wire] = frozenset()
    
    def elapsed_ok():
        return time.time() < deadline
    
    # MST-inspired approach: Build a priority queue of candidate pairs
    # weighted by how many rows they appear in (frequency), with tie-breaking
    # by popcount of resulting expression and usage count
    
    # First, compute initial pair frequencies across all rows
    def compute_pair_frequencies():
        """Compute frequency of each wire pair appearing together in rows."""
        pair_freq = defaultdict(int)
        wire_usage = defaultdict(int)
        
        for i in range(n_unique):
            rs = row_wires_unique[i]
            if len(rs) < 2:
                continue
            elems = sorted(rs)
            for w in elems:
                wire_usage[w] += 1
            for x in range(len(elems)):
                for y in range(x + 1, len(elems)):
                    a, b = elems[x], elems[y]
                    pair_freq[(a, b)] += 1
        
        return pair_freq, wire_usage
    
    # Main greedy loop with MST-inspired priority
    max_iter = 100000
    for _ in range(max_iter):
        if not elapsed_ok():
            break
        
        # Compute current pair frequencies
        pair_freq, wire_usage = compute_pair_frequencies()
        
        if not pair_freq:
            break
        
        # Build priority queue: prioritize pairs by frequency (benefit = freq - 1)
        # Tie-break: lower popcount of new expression, then higher combined usage
        candidates = []
        for (a, b), freq in pair_freq.items():
            if freq >= 2:
                benefit = freq - 1
                new_set = wire_set[a] ^ wire_set[b]
                new_pop = len(new_set)
                usage = wire_usage.get(a, 0) + wire_usage.get(b, 0)
                # Negate benefit for min-heap, then use popcount and -usage for tie-breaking
                candidates.append((-benefit, new_pop, -usage, a, b))
        
        if not candidates:
            break
        
        # Use heap to find best candidate efficiently
        heapq.heapify(candidates)
        best = heapq.heappop(candidates)
        
        # Check if this pair is still valid (both wires still in at least 2 rows)
        a, b = best[3], best[4]
        # Verify the pair still has sufficient frequency
        if (a, b) not in pair_freq or pair_freq[(a, b)] < 2:
            continue
        
        new_wire = next_wire
        next_wire += 1
        
        # Compute the set for the new wire
        new_set = wire_set[a] ^ wire_set[b]
        wire_set[new_wire] = new_set
        gates.append([a, b])
        
        # Update all unique rows containing both a and b
        for i in range(n_unique):
            rs = row_wires_unique[i]
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                rs.add(new_wire)
    
    # Build outputs for unique rows
    unique_outputs = []
    for i in range(n_unique):
        rs = row_wires_unique[i]
        if len(rs) == 0:
            unique_outputs.append(zero_wire)
        elif len(rs) == 1:
            unique_outputs.append(list(rs)[0])
        else:
            elems = list(rs)
            current = elems[0]
            for e in elems[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            unique_outputs.append(current)
            
    # Map back to original rows
    outputs = []
    for i in range(n_rows):
        outputs.append(unique_outputs[row_to_unique[i]])
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
