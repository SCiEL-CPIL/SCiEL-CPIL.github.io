"""XOR-circuit synthesis: minimize mean gate-count / independent-row baseline."""
import time
import numpy as np
from collections import defaultdict

# We treat the matrix as a binary matrix over GF(2).
# Input wires: 0..n-1 (columns). Constant-zero wire: n.
# Gate k produces wire n+1+k.
# Each row i defines a target vector v_i in GF(2)^n.
# We need to express each v_i as an XOR-sum of a subset of input wires.
# The "independent-row baseline" for row i is (popcount(v_i) - 1) gates if popcount>1, else 0.
# Actually, to produce a wire that is the XOR of k inputs, you need k-1 gates.
# Baseline total = sum over rows of max(0, popcount(row) - 1).
# We want to minimize total gates by sharing intermediate XORs.

def _row_baseline(matrix):
    """Compute the independent-row baseline gate count."""
    # matrix is expected to be a 2D array-like of 0s and 1s
    m = np.asarray(matrix, dtype=np.uint8)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    popcounts = m.sum(axis=1)
    gates = np.maximum(0, popcounts - 1).astype(np.int64)
    return int(gates.sum())

def _verify(matrix, gates, outputs, depth_limit):
    """
    Exact symbolic GF(2) verification.
    matrix: 2D array, rows are targets.
    gates: list of [wire_a, wire_b]
    outputs: list of wire indices, one per row.
    Returns (valid, max_depth).
    """
    m = np.asarray(matrix, dtype=np.uint8)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n_rows, n_cols = m.shape
    n = n_cols
    zero_wire = n
    # Each wire corresponds to a vector in GF(2)^n.
    # Input wire i -> e_i (unit vector)
    # Constant-zero wire n -> zero vector
    # Gate wire n+1+k -> XOR of its two inputs.
    
    wire_vec = {}
    wire_depth = {}
    
    for i in range(n):
        v = np.zeros(n, dtype=np.uint8)
        v[i] = 1
        wire_vec[i] = v
        wire_depth[i] = 0
    
    wire_vec[zero_wire] = np.zeros(n, dtype=np.uint8)
    wire_depth[zero_wire] = 0
    
    max_gate_depth = 0
    for k, (a, b) in enumerate(gates):
        wire_id = n + 1 + k
        if a not in wire_vec or b not in wire_vec:
            return False, 0
        v = (wire_vec[a] ^ wire_vec[b]).astype(np.uint8)
        wire_vec[wire_id] = v
        d = max(wire_depth[a], wire_depth[b]) + 1
        wire_depth[wire_id] = d
        if d > max_gate_depth:
            max_gate_depth = d
    
    # Check outputs
    if len(outputs) != n_rows:
        return False, 0
    
    for i in range(n_rows):
        w = outputs[i]
        if w not in wire_vec:
            return False, 0
        if not np.array_equal(wire_vec[w], m[i]):
            return False, 0
        if wire_depth[w] > depth_limit:
            return False, 0
    
    return True, max_gate_depth

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    matrix: 2D binary array (rows are targets, columns are inputs)
    depth_limit: max circuit depth allowed (<=16 per contract, but we respect the argument)
    Returns: {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    """
    t_start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    m = np.asarray(matrix, dtype=np.uint8)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n_rows, n_cols = m.shape
    n = n_cols
    zero_wire = n
    
    # Baseline
    baseline = _row_baseline(m)
    
    # We'll use a greedy set-cover / reuse approach.
    # Strategy:
    # 1. For each row, we need to build its vector from inputs.
    # 2. We maintain a dictionary of available vectors -> wire_id.
    # 3. Initially, inputs and zero are available.
    # 4. For each row, find a subset of available vectors that XOR to the target,
    #    minimizing the number of NEW gates needed.
    #    If no subset is found with 0 new gates, try to find one with minimal new gates.
    #    A simple heuristic: express the target as XOR of a small number of available vectors.
    #    This is a set cover problem, which is NP-hard, but for small n and limited depth,
    #    we can use a greedy approach or BFS on the vector space.
    
    # Actually, a better approach for XOR circuits:
    # The vectors form a vector space over GF(2). We want to build each row vector
    # using the fewest gates, sharing intermediates.
    # 
    # Key insight: if we have a set of available vectors, and we want to build a target,
    # we can XOR two available vectors to get a new one (cost 1 gate), then use that.
    # This is like finding a path in the Cayley graph of the vector space.
    #
    # For small n (<=16), we can use BFS to find the shortest sequence of XORs
    # from the available set to the target. But the "available set" grows as we add gates.
    #
    # Simpler greedy: process rows in order of increasing popcount (easier first).
    # For each row, try to express it as XOR of <= 2 available vectors (cost 0 or 1 gate).
    # If not possible, build it from inputs directly, adding intermediate gates.
    
    # Let's use a different approach: 
    # 1. Start with all input vectors and zero available.
    # 2. For each target row, find the minimal-cost way to produce it, where cost
    #    is the number of new gates. We allow using any combination of available vectors.
    # 3. To find a low-cost expression, we can use the fact that XOR of k vectors
    #    can be done in k-1 gates. So if we can express the target as XOR of k available
    #    vectors, cost is k-1 (but some of those vectors might not be "single" wires).
    #
    # Actually, each available vector IS a single wire. So XOR of k available wires
    # takes k-1 gates. But we can also create new intermediate wires.
    #
    # Let's use a greedy approach with reuse:
    # - available: dict mapping frozenset-of-inputs (or vector) to wire_id
    # - For each row, find a subset S of available vectors such that XOR(S) = target.
    #   Minimize |S| - 1 (number of new gates to combine them).
    #   If |S| = 1, cost 0. If |S| = 2, cost 1. Etc.
    # - If no such S exists with small |S|, build the target from inputs.
    
    # For efficiency, represent vectors as integers (bitmasks) for n <= 64.
    if n > 60:
        # Fallback to baseline for large n
        gates = []
        outputs = []
        wire_counter = n + 1
        for i in range(n_rows):
            row = m[i]
            indices = np.nonzero(row)[0]
            if len(indices) == 0:
                outputs.append(zero_wire)
            elif len(indices) == 1:
                outputs.append(indices[0])
            else:
                # Build from inputs
                acc = indices[0]
                for idx in indices[1:]:
                    gates.append([acc, idx])
                    acc = wire_counter
                    wire_counter += 1
                outputs.append(acc)
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n):
            if m[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Available wires: mask -> wire_id
    available = {}
    # Input wires
    for j in range(n):
        available[1 << j] = j
    # Zero wire
    available[0] = zero_wire
    
    gates = []
    wire_counter = n + 1
    
    # Process rows: sort by popcount (ascending) to build simpler ones first
    row_order = sorted(range(n_rows), key=lambda i: bin(row_masks[i]).count('1'))
    
    outputs = [0] * n_rows
    
    for idx in row_order:
        target = row_masks[idx]
        
        # Try to find a small subset of available masks that XOR to target.
        # Try subset sizes 1, 2, 3, ... up to some limit.
        found = False
        best_subset = None
        best_cost = float('inf')
        
        avail_masks = list(available.keys())
        # Limit search for efficiency
        max_subset_size = 4
        
        for size in range(1, max_subset_size + 1):
            if found:
                break
            # Try all combinations of size from avail_masks
            # This is expensive, so limit avail_masks size
            if len(avail_masks) > 200:
                # Sample or use a heuristic
                # For now, skip large searches
                break
            from itertools import combinations
            for combo in combinations(avail_masks, size):
                xor_result = 0
                for c in combo:
                    xor_result ^= c
                if xor_result == target:
                    cost = size - 1
                    if cost < best_cost:
                        best_cost = cost
                        best_subset = combo
                        found = True
                    if cost == 0:
                        break
            if found and best_cost == 0:
                break
        
        if found and best_subset is not None:
            if best_cost == 0:
                outputs[idx] = available[best_subset[0]]
            else:
                # Combine the subset with best_cost new gates
                acc = available[best_subset[0]]
                for c in best_subset[1:]:
                    gates.append([acc, available[c]])
                    acc = wire_counter
                    wire_counter += 1
                    available[target] = acc  # The result is the target
                outputs[idx] = acc
        else:
            # Build from inputs directly
            indices = [j for j in range(n) if (target >> j) & 1]
            if len(indices) == 0:
                outputs[idx] = zero_wire
            elif len(indices) == 1:
                outputs[idx] = indices[0]
            else:
                acc = indices[0]
                for j in indices[1:]:
                    gates.append([acc, j])
                    acc = wire_counter
                    wire_counter += 1
                outputs[idx] = acc
                available[target] = acc
        
        # Check deadline
        if time.time() - t_start > DEADLINE:
            break
    
    # Fill any unprocessed rows with direct construction
    for idx in range(n_rows):
        if outputs[idx] == 0 and row_masks[idx] != 0:
            # Might not have been processed
            target = row_masks[idx]
            indices = [j for j in range(n) if (target >> j) & 1]
            if len(indices) == 0:
                outputs[idx] = zero_wire
            elif len(indices) == 1:
                outputs[idx] = indices[0]
            else:
                acc = indices[0]
                for j in indices[1:]:
                    gates.append([acc, j])
                    acc = wire_counter
                    wire_counter += 1
                outputs[idx] = acc
    
    # Verify
    valid, max_depth = _verify(m, gates, outputs, depth_limit)
    if not valid:
        # Fallback to pure baseline
        gates = []
        outputs = []
        wire_counter = n + 1
        for i in range(n_rows):
            row = m[i]
            indices = np.nonzero(row)[0]
            if len(indices) == 0:
                outputs.append(zero_wire)
            elif len(indices) == 1:
                outputs.append(indices[0])
            else:
                acc = indices[0]
                for idx in indices[1:]:
                    gates.append([acc, idx])
                    acc = wire_counter
                    wire_counter += 1
                outputs.append(acc)
    
    return {'gates': gates, 'outputs': outputs}
