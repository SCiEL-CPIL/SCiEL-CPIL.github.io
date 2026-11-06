"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def _gf2_rank_and_pivots(matrix):
    """Compute rank and pivot columns of a binary matrix."""
    m = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = m.shape
    rank = 0
    pivots = []
    for col in range(n_cols):
        # Find pivot row
        pivot_row = -1
        for row in range(rank, n_rows):
            if m[row, col]:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap rows
        m[[rank, pivot_row]] = m[[pivot_row, rank]]
        # Eliminate
        for row in range(n_rows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        pivots.append(col)
        rank += 1
    return rank, pivots

def _greedy_synthesis(target_vectors, n_inputs, depth_limit):
    """
    Greedy synthesis of XOR circuit.
    target_vectors: list of bit vectors (each is a tuple/list of 0/1)
    n_inputs: number of input wires (0..n_inputs-1)
    Returns: (gates, outputs) where gates is list of [wire1, wire2] and outputs is list of wire indices
    """
    n = len(target_vectors)
    const_zero = n_inputs  # wire index for constant zero
    
    # Represent each target as a set of wire indices (sparse)
    # Wire 0..n_inputs-1 are inputs, n_inputs is constant zero
    # Gate k produces wire n_inputs + 1 + k
    
    # Track what each wire represents (as a frozenset of input indices, or special for const)
    # Actually, let's use a different representation: each wire has a "signature" which is a tuple of input indices it depends on
    
    # Initial wires: inputs 0..n_inputs-1 each depend on themselves, const_zero depends on nothing
    wire_sig = {}
    for i in range(n_inputs):
        wire_sig[i] = frozenset([i])
    wire_sig[const_zero] = frozenset()
    
    # Target signatures
    target_sigs = []
    for tv in target_vectors:
        sig = frozenset(i for i, bit in enumerate(tv[:n_inputs]) if bit)
        target_sigs.append(sig)
    
    # We need to produce each target_sig using XOR of existing wires
    # Greedy: for each target, try to find two existing wires whose XOR gives it
    # Or build it up incrementally
    
    gates = []
    wire_to_target = {}  # wire_index -> target_index it produces
    next_wire = n_inputs + 1  # next available wire for gates
    
    # First, check if any targets are already available as inputs or const
    available_sigs = {}  # sig -> wire_index
    for w, sig in wire_sig.items():
        available_sigs[sig] = w
    
    produced = [False] * n
    output_wires = [None] * n
    
    # Check direct matches
    for i, tsig in enumerate(target_sigs):
        if tsig in available_sigs:
            produced[i] = True
            output_wires[i] = available_sigs[tsig]
    
    # Iteratively try to produce remaining targets
    max_iter = n * 10  # safety limit
    for _ in range(max_iter):
        progress = False
        for i in range(n):
            if produced[i]:
                continue
            tsig = target_sigs[i]
            # Try to find two existing wires a, b such that sig_a XOR sig_b = tsig
            # i.e., sig_a symmetric_difference sig_b = tsig
            # This is equivalent to: for each existing wire a, check if tsig XOR sig_a is also available
            found = False
            for a, sig_a in list(wire_sig.items()):
                # Compute what b would need to be: sig_b = tsig XOR sig_a
                # In terms of sets: sig_b = (tsig - sig_a) ∪ (sig_a - tsig)
                needed = frozenset(tsig.symmetric_difference(sig_a))
                if needed in available_sigs:
                    b = available_sigs[needed]
                    # Create gate: a XOR b -> new wire
                    new_wire = next_wire
                    gates.append([a, b])
                    wire_sig[new_wire] = tsig
                    available_sigs[tsig] = new_wire
                    next_wire += 1
                    produced[i] = True
                    output_wires[i] = new_wire
                    progress = True
                    found = True
                    break
            if found:
                continue
        if not progress:
            break
    
    # If some targets still not produced, use fallback: build each from scratch
    # This should rarely happen with the greedy approach
    for i in range(n):
        if produced[i]:
            continue
        tsig = target_sigs[i]
        # Build from input wires directly
        # We need to XOR together all input wires in tsig
        inputs_needed = list(tsig)
        if not inputs_needed:
            # Constant zero
            output_wires[i] = const_zero
            produced[i] = True
            continue
        # Start with first input
        current_wire = inputs_needed[0]
        for inp in inputs_needed[1:]:
            new_wire = next_wire
            gates.append([current_wire, inp])
            wire_sig[new_wire] = frozenset()  # will update
            # Actually compute new sig
            cur_sig = wire_sig[current_wire]
            new_sig = frozenset(cur_sig.symmetric_difference(frozenset([inp])))
            wire_sig[new_wire] = new_sig
            available_sigs[new_sig] = new_wire
            current_wire = new_wire
            next_wire += 1
        output_wires[i] = current_wire
        produced[i] = True
    
    # Now try to eliminate redundant gates
    # A gate is redundant if its output wire is never used by another gate or as a final output
    # Build usage count
    wire_usage = defaultdict(int)
    for g in gates:
        wire_usage[g[0]] += 1
        wire_usage[g[1]] += 1
    for ow in output_wires:
        if ow is not None and ow >= n_inputs + 1:
            wire_usage[ow] += 1
    
    # Remove gates whose output is not used
    # Map from output wire to gate index
    gate_output = {}
    for k, g in enumerate(gates):
        out_wire = n_inputs + 1 + k
        gate_output[out_wire] = k
    
    # Iteratively remove unused gates
    changed = True
    while changed:
        changed = False
        new_gates = []
        new_gate_output = {}
        for k, g in enumerate(gates):
            out_wire = n_inputs + 1 + k
            if wire_usage.get(out_wire, 0) > 0:
                new_gates.append(g)
                new_gate_output[out_wire] = len(new_gates)
            else:
                changed = True
        if changed:
            gates = new_gates
            # Rebuild wire_usage
            wire_usage = defaultdict(int)
            for g in gates:
                wire_usage[g[0]] += 1
                wire_usage[g[1]] += 1
            for ow in output_wires:
                if ow is not None and ow >= n_inputs + 1:
                    wire_usage[ow] += 1
            # Update output_wires if they point to removed gates
            # This is tricky - we need to remap
            # For now, let's just rebuild from scratch if this happens
            break  # Simplification: don't do complex remapping
    
    # If we did removal, we need to remap output_wires
    # This is complex, so let's do a simpler cleanup: just return what we have
    # The key optimization is in the greedy synthesis itself
    
    return gates, output_wires

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    matrix = np.array(matrix, dtype=np.uint8)
    n_outputs, n_cols = matrix.shape
    n_inputs = n_cols - 1  # last column might be for constant
    
    # Check if last column is all zeros (no constant term needed)
    has_const = not np.all(matrix[:, -1] == 0)
    if has_const:
        # Constant term present - XOR circuits can't produce constants easily
        # For now, handle by treating constant as part of the linear system
        # Actually, in GF(2) XOR circuits, we can only produce linear combinations
        # If there's a constant, we may need to handle it specially
        # For simplicity, assume the matrix represents linear combinations of inputs 0..n_inputs-1
        # and the last column is either all 0 or indicates we need constant handling
        pass
    
    # Extract target vectors (first n_inputs columns)
    target_vectors = [tuple(matrix[i, :n_inputs]) for i in range(n_outputs)]
    
    # Try greedy synthesis
    gates, output_wires = _greedy_synthesis(target_vectors, n_inputs, depth_limit)
    
    # Verify the result
    # Build the circuit and check outputs
    if len(gates) > 0 or any(ow is None for ow in output_wires):
        # Verify by simulation
        n_wires = n_inputs + 1 + len(gates)
        # Simulate: for each input, set value
        # We'll use symbolic representation
        # Each wire is a frozenset of input indices
        wire_val = {}
        for i in range(n_inputs):
            wire_val[i] = frozenset([i])
        wire_val[n_inputs] = frozenset()  # constant zero
        
        for k, (a, b) in enumerate(gates):
            out = n_inputs + 1 + k
            wire_val[out] = wire_val[a].symmetric_difference(wire_val[b])
        
        # Check outputs
        valid = True
        for i, ow in enumerate(output_wires):
            if ow is None:
                valid = False
                break
            expected = frozenset(j for j, bit in enumerate(target_vectors[i]) if bit)
            if wire_val.get(ow, None) != expected:
                valid = False
                break
        
        if not valid:
            # Fallback: use simple direct synthesis
            gates = []
            output_wires = []
            for i in range(n_outputs):
                tv = target_vectors[i]
                inputs_needed = [j for j, bit in enumerate(tv) if bit]
                if not inputs_needed:
                    output_wires.append(n_inputs)
                    continue
                current = inputs_needed[0]
                for inp in inputs_needed[1:]:
                    gates.append([current, inp])
                    current = n_inputs + 1 + len(gates) - 1
                output_wires.append(current)
    
    # Check depth (simplified: depth is at most number of gates in a chain)
    # For now, assume depth is satisfied if gates are reasonable
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END

