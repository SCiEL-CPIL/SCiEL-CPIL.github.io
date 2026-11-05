"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s kill
    
    # Convert to numpy array
    mat = np.array(matrix, dtype=np.int8)
    n_rows, n_cols = mat.shape
    
    # Interpretation: each column is an output function
    # mat[i, j] = 1 means input i is XORed into output j
    # So output j = XOR of inputs i where mat[i,j] == 1
    
    # Build the set of input indices for each output
    output_input_sets = []
    for j in range(n_cols):
        inputs = set()
        for i in range(n_rows):
            if mat[i, j]:
                inputs.add(i)
        output_input_sets.append(inputs)
    
    # Find pairs of inputs that appear in multiple outputs
    pair_outputs = {}  # frozenset -> set of output indices
    for j, inputs in enumerate(output_input_sets):
        if len(inputs) < 2:
            continue
        input_list = sorted(inputs)
        for i1 in range(len(input_list)):
            for i2 in range(i1 + 1, len(input_list)):
                pair = frozenset([input_list[i1], input_list[i2]])
                if pair not in pair_outputs:
                    pair_outputs[pair] = set()
                pair_outputs[pair].add(j)
    
    # Select pairs that appear in >= 2 outputs as shared subexpressions
    # Sort by frequency descending
    shared_pairs = {}  # frozenset -> wire number
    next_wire = n_rows + 1  # wire n_rows is constant zero
    
    for pair, outputs in sorted(pair_outputs.items(), key=lambda x: -len(x[1])):
        if len(outputs) >= 2:
            shared_pairs[pair] = next_wire
            next_wire += 1
    
    # Build the circuit
    gates = []
    
    # First, add gates for shared pairs
    for pair, wire in shared_pairs.items():
        a, b = list(pair)
        gates.append([a, b])
    
    output_wires = []
    
    for j, inputs in enumerate(output_input_sets):
        if not inputs:
            # Output is 0
            output_wires.append(n_rows)
            continue
        
        if len(inputs) == 1:
            # Just one input
            output_wires.append(list(inputs)[0])
            continue
        
        # Try to use shared subexpressions
        remaining = set(inputs)
        used_wires = []
        
        # For each shared pair, if both inputs are in remaining, use it
        for pair, wire in shared_pairs.items():
            if pair.issubset(remaining):
                used_wires.append(wire)
                remaining -= pair
        
        # Add any remaining individual inputs
        for inp in sorted(remaining):
            used_wires.append(inp)
        
        # Now compute XOR of used_wires
        if not used_wires:
            output_wires.append(n_rows)
        elif len(used_wires) == 1:
            output_wires.append(used_wires[0])
        else:
            # Build XOR tree
            used_wires.sort()
            current = used_wires[0]
            for w in used_wires[1:]:
                gates.append([current, w])
                current = next_wire
                next_wire += 1
            output_wires.append(current)
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
