"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import random
import sys
import heapq

sys.setrecursionlimit(100000)

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix using MST-based
    shared subexpression extraction on the hypercube graph.
    
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
    
    # Represent each row as a frozenset of input indices
    row_fsets = []
    for i in range(n_rows):
        row_fsets.append(frozenset(np.where(A[i] == 1)[0].tolist()))
    
    def elapsed_ok():
        return time.time() < deadline
    
    def mst_synthesis():
        """
        MST-based XOR circuit synthesis.
        
        Model: Each possible intermediate wire corresponds to a subset S of inputs.
        Creating wire S from wires A and B (where A XOR B = S) costs 1 gate.
        The benefit of creating wire S is the number of target rows that contain S
        as a sub-expression (i.e., can use S to reduce their remaining work).
        
        We build a graph where:
        - Nodes are "states" (subsets of inputs)
        - Edges represent XOR operations that create a new state from two existing ones
        - Edge weight = -benefit (we want to maximize benefit, so minimize negative)
        
        Kruskal's algorithm selects edges in order of benefit, ensuring we don't
        create redundant wires (tree structure).
        """
        # Identify all relevant subsets: subsets of target rows that are "useful"
        # A subset S is useful if it appears in at least 2 target rows (as a sub-multiset)
        # or is a target row itself
        
        # For efficiency, we'll consider subsets up to a certain size
        max_subset_size = min(8, n_cols)  # Limit subset size for tractability
        
        # Count frequency of each subset across all target rows
        # A subset S "appears" in row R if S is a subset of R
        # But we need to be careful: we want to count how many rows would benefit from having S precomputed
        
        # Simpler approach: For each pair of elements that co-occur in rows,
        # count how many rows contain both. The benefit of creating a wire for {a,b}
        # is (count - 1) because we save one gate per additional row beyond the first.
        
        # Generalize: for subset S of size k, the benefit is (freq(S) - 1) if freq(S) >= 2,
        # where freq(S) = number of target rows that contain S as a subset.
        
        # But computing freq for all subsets is expensive. Let's use a greedy approach
        # that builds up from pairs.
        
        # Track current state of each row: which "wires" (subsets) it still needs
        row_wires = [set(fs) for fs in row_fsets]
        
        # Map from frozenset to wire index
        set_to_wire = {}
        for i in range(n_inputs):
            set_to_wire[frozenset([i])] = i
        set_to_wire[frozenset()] = zero_wire
        
        gates = []
        next_wire = zero_wire + 1
        wire_depth = {i: 0 for i in range(n_inputs)}
        wire_depth[zero_wire] = 0
        wire_set = {i: frozenset([i]) for i in range(n_inputs)}
        wire_set[zero_wire] = frozenset()
        
        # Priority queue of candidate merges: (neg_benefit, wire_a, wire_b)
        # We'll recompute benefits periodically
        
        def compute_pair_benefits():
            """Compute benefit for each pair of wires that appear together in rows."""
            pair_freq = defaultdict(int)
            for i in range(n_rows):
                rs = row_wires[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a == zero_wire or b == zero_wire:
                            continue
                        key = (min(a, b), max(a, b))
                        pair_freq[key] += 1
            return pair_freq
        
        max_iter = 20000
        for iteration in range(max_iter):
            if not elapsed_ok():
                break
            
            pair_freq = compute_pair_benefits()
            
            if not pair_freq:
                break
            
            # Find the pair with highest benefit (freq - 1)
            best_pair = None
            best_benefit = 0
            for (a, b), freq in pair_freq.items():
                if freq < 2:
                    continue
                benefit = freq - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            
            # Create new wire
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_depth[new_wire] = max(wire_depth.get(a, 0), wire_depth.get(b, 0)) + 1
            
            # Compute the set represented by the new wire
            set_a = wire_set.get(a, frozenset())
            set_b = wire_set.get(b, frozenset())
            new_set = frozenset(set_a ^ set_b)  # Symmetric difference = XOR
            wire_set[new_wire] = new_set
            set_to_wire[new_set] = new_wire
            
            # Update rows: replace a and b with new_wire where both are present
            for i in range(n_rows):
                rs = row_wires[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Final pass: resolve remaining multi-wire rows
        outputs = []
        for i in range(n_rows):
            rs = row_wires[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = sorted(rs, key=lambda w: wire_depth.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wire_depth.get(current, 0), wire_depth.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        # Fallback: just use a linear chain without depth check
                        pass
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wire_depth[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    def greedy_synthesis():
        """Standard greedy synthesis as fallback."""
        gates = []
        next_wire = zero_wire + 1
        
        row_wires = [set(fs) for fs in row_fsets]
        
        set_to_wire = {}
        for i in range(n_inputs):
            set_to_wire[frozenset([i])] = i
        set_to_wire[frozenset()] = zero_wire
        
        wire_depth = {i: 0 for i in range(n_inputs)}
        wire_depth[zero_wire] = 0
        
        max_iter = 50000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = row_wires[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a == zero_wire or b == zero_wire:
                            continue
                        key = (min(a, b), max(a, b))
                        pair_count[key] += 1
            
            if not pair_count:
                break
            
            best_pair = None
            best_benefit = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_depth[new_wire] = max(wire_depth.get(a, 0), wire_depth.get(b, 0)) + 1
            
            for i in range(n_rows):
                rs = row_wires[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        outputs = []
        for i in range(n_rows):
            rs = row_wires[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = sorted(rs, key=lambda w: wire_depth.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wire_depth.get(current, 0), wire_depth.get(e, 0)) + 1
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wire_depth[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try MST-based synthesis first
    best_result = None
    best_gate_count = float('inf')
    
    if elapsed_ok():
        result = mst_synthesis()
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    # Try greedy as well
    if elapsed_ok():
        result = greedy_synthesis()
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    if best_result is None:
        # Fallback: independent row synthesis
        gates = []
        next_wire = zero_wire + 1
        outputs = []
        for i in range(n_rows):
            rs = row_fsets[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                current = elems[0]
                for e in elems[1:]:
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        return {'gates': gates, 'outputs': outputs}
    
    return best_result

# EVOLVE-BLOCK-END
