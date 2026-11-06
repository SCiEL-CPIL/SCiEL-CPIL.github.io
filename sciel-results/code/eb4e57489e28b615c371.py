"""
Exact XOR-circuit synthesis using global Common Subexpression Elimination (CSE).
This approach treats the synthesis as a global optimization problem where we 
iteratively select the most beneficial intermediate XOR to create, maximizing 
shared subexpressions across all output rows simultaneously.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """
    Verify that the given gates and outputs correctly compute matrix @ x over GF(2).
    """
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    
    if depth_limit is not None:
        depth = [0] * total_wires
        for k in range(num_gates):
            a, b = gates[k]
            if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
                return False
            if a >= n + 1 + k or b >= n + 1 + k:
                return False
            depth[n + 1 + k] = 1 + max(depth[a], depth[b])
        max_depth = max(depth[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
        for wire in outputs:
            if wire < 0 or wire >= total_wires:
                return False
            max_depth = max(max_depth, depth[wire])
        if max_depth > depth_limit:
            return False
    
    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    
    for i in range(m):
        wire = outputs[i]
        if wire < 0 or wire >= total_wires:
            return False
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    
    return True

def _baseline_synthesis(matrix, depth_limit=16):
    """
    Baseline synthesis: independent XOR chains for each row.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    
    for i in range(m):
        row = matrix[i]
        positions = np.where(row)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([a, b])
                current = new_wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

def _cse_synthesis(matrix, depth_limit=16, time_budget=40.0):
    """
    Global Common Subexpression Elimination (CSE) synthesis.
    
    Strategy:
    1. Represent each output row as a set of input indices.
    2. Maintain a set of "available" wires (inputs + zero + created gates).
    3. Iteratively find the XOR pair (u, v) of available wires such that the 
       new wire w = u ^ v provides the maximum "savings" in total gate count 
       when used to simplify the representation of the output rows.
    4. The "savings" are estimated by counting how many output rows can be 
       simplified by using w (i.e., if a row contains both u and v, replacing 
       them with w saves 1 gate in that row's chain).
    5. Continue until no more beneficial XORs are found or time runs out.
    6. Finally, construct the gate list by tracing the dependency tree of 
       the simplified output representations.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Represent each row as a set of input indices
    row_sets = [set(np.where(matrix[i])[0].tolist()) for i in range(m)]
    
    # Available wires: wire_id -> set of input indices
    # Inputs are 0..n-1, zero is n
    available_wires = {}
    for i in range(n):
        available_wires[i] = {i}
    available_wires[n] = set()
    
    # Map from frozenset to wire_id (for quick lookup)
    set_to_wire = {}
    for w, s in available_wires.items():
        set_to_wire[frozenset(s)] = w
    
    # List of created gates: [wire_a, wire_b] -> wire_c
    gates = []
    
    # Iteratively create new wires
    max_iterations = 10000
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        
        best_savings = 0
        best_pair = None
        best_new_set = None
        
        # Get list of current wire IDs to iterate
        wire_ids = list(available_wires.keys())
        num_wires = len(wire_ids)
        
        # To avoid O(W^2) explosion, we can limit the search or use heuristics.
        # For small n (8-16), W might grow to ~50-100. 100^2 = 10000 pairs.
        # 10000 * m (16) * set_ops is manageable.
        
        for i in range(num_wires):
            w1 = wire_ids[i]
            s1 = available_wires[w1]
            for j in range(i + 1, num_wires):
                w2 = wire_ids[j]
                s2 = available_wires[w2]
                
                # New set
                s_new = s1.symmetric_difference(s2)
                
                # Check if already exists
                if frozenset(s_new) in set_to_wire:
                    continue
                
                # Estimate savings
                # A new wire w = s1 ^ s2 is useful if it helps simplify some rows.
                # Specifically, if a row R contains s1 and s2, we can replace 
                # s1, s2 in R with w. This reduces the size of R's set by 1 
                # (since |s1 ^ s2| = |s1| + |s2| - 2*|s1 & s2|, and we remove 2 
                # elements and add 1, net change -1 if s1 and s2 are disjoint? 
                # No, the cost is the number of XORs needed to build the set.
                # Cost to build a set S from inputs is roughly |S|-1 (chain) or 
                # less if shared.
                
                # Let's define "cost" of a row R as the minimum number of XORs 
                # needed to build it from available wires.
                # If R contains w, cost(R) might decrease.
                
                # Heuristic: Count how many rows have both s1 and s2 as subsets?
                # No, that's not quite right. The row is a set of inputs.
                # If we introduce w, a row R can use w if w is a subset of R?
                # No, w is a vector. R is a vector.
                # R = w ^ (R ^ w).
                # So if we have w, we need to build R ^ w.
                # The savings is: cost(R) - 1 - cost(R ^ w).
                # If cost(R) is approximated by |R|-1 (chain), and cost(R^w) by |R^w|-1,
                # then savings = (|R|-1) - 1 - (|R^w|-1) = |R| - |R^w| - 1.
                # |R^w| = |R| + |w| - 2*|R & w|.
                # Savings = |R| - (|R| + |w| - 2*|R & w|) - 1 = 2*|R & w| - |w| - 1.
                
                # Let's use this heuristic.
                total_savings = 0
                for R in row_sets:
                    # R is a set of input indices
                    # w is s_new
                    # Intersection
                    inter_size = len(R.intersection(s_new))
                    w_size = len(s_new)
                    savings = 2 * inter_size - w_size - 1
                    if savings > 0:
                        total_savings += savings
                
                if total_savings > best_savings:
                    best_savings = total_savings
                    best_pair = (w1, w2)
                    best_new_set = s_new
        
        if best_pair is None:
            break
        
        # Create the new wire
        w1, w2 = best_pair
        new_wire_id = n + 1 + len(gates)
        gates.append([w1, w2])
        available_wires[new_wire_id] = best_new_set
        set_to_wire[frozenset(best_new_set)] = new_wire_id
        
        iter_count += 1
    
    # Now, construct the final outputs using the available wires.
    # For each row, find the best representation using available wires.
    # We want to minimize the number of XORs.
    # This is a set cover-like problem, but we can use a greedy approach.
    
    outputs = [None] * m
    # We need to determine which wires are actually used in the final circuit.
    # We'll build the circuit by resolving each row.
    
    # For each row, find the sequence of XORs to build it from inputs/zero.
    # We can use BFS from the row's target set to the inputs, using available wires.
    
    # Actually, since we created gates greedily, we can just trace back.
    # But the gates are a DAG. We need to ensure all gates used in outputs are included.
    
    # Let's use a simple resolver:
    # For each row, find the minimum XOR expression using the available wires.
    
    # We'll use a BFS from the target row set, expanding by XOR-ing with available wires.
    # State: current set. Goal: reach a set that is a single input or zero.
    # But we can also reach any available wire.
    
    # Let's precompute the "cost" to build each available wire from inputs?
    # No, the cost is 1 per gate.
    
    # Simpler: For each row, find the best single wire that is a subset? No.
    # The row is a vector. We can express it as XOR of available wires.
    
    # Let's use a greedy set cover for each row:
    # While row_set is not empty:
    #   Find the available wire w that maximizes |row_set & w| / (cost of w)
    #   XOR w into row_set
    #   Add w to the list of wires used for this row
    
    # But we need to output the actual gate indices.
    # So we need to trace the dependencies of the chosen wires.
    
    # Let's do this:
    # 1. For each row, find the sequence of available wires to XOR together.
    # 2. Collect all wires used.
    # 3. Topologically sort the gates to ensure dependencies are met.
    # 4. Filter gates to only those needed.
    # 5. Assign final wire indices.
    
    # Step 1: Resolve each row
    row_wires = []  # List of lists of wire IDs
    for i in range(m):
        current_set = set(row_sets[i])
        wires_used = []
        
        while current_set:
            best_w = None
            best_score = -1
            
            for w_id, w_set in available_wires.items():
                if not w_set:
                    continue
                inter = len(current_set.intersection(w_set))
                if inter == 0:
                    continue
                # Score: how much it reduces the set size?
                # |current ^ w| = |current| + |w| - 2*inter
                # Reduction = |current| - |current ^ w| = 2*inter - |w|
                reduction = 2 * inter - len(w_set)
                if reduction > best_score:
                    best_score = reduction
                    best_w = w_id
            
            if best_w is None:
                # Should not happen if inputs are available
                break
            
            wires_used.append(best_w)
            current_set = current_set.symmetric_difference(available_wires[best_w])
        
        row_wires.append(wires_used)
    
    # Step 2: Collect all used wires
    used_wires = set()
    for wires in row_wires:
        used_wires.update(wires)
    
    # Step 3: Determine which gates are needed
    # A gate k produces wire n+1+k. It is needed if its output is in used_wires
    # or if its output is needed to produce another used wire.
    
    needed_gates = set()
    # Reverse dependency: gate k depends on wires a, b.
    # If wire n+1+k is needed, then gate k is needed.
    # If wire a is produced by a gate, and wire a is needed, then that gate is needed.
    
    # Let's build a map: wire_id -> gate_index (if produced by a gate)
    wire_to_gate = {}
    for k, (a, b) in enumerate(gates):
        wire_to_gate[n + 1 + k] = k
    
    # BFS from used_wires to find all needed gates
    to_visit = list(used_wires)
    while to_visit:
        w = to_visit.pop()
        if w in wire_to_gate:
            k = wire_to_gate[w]
            if k not in needed_gates:
                needed_gates.add(k)
                a, b = gates[k]
                to_visit.append(a)
                to_visit.append(b)
    
    # Step 4: Rebuild the gate list with only needed gates
    # We need to re-index the wires.
    # Original wire IDs: 0..n-1 (inputs), n (zero), n+1.. (gates)
    # New wire IDs: 0..n-1 (inputs), n (zero), n+1.. (new gates)
    
    # Map old wire ID to new wire ID
    old_to_new = {i: i for i in range(n + 1)}
    new_gates = []
    
    # Sort needed gates by original index to maintain topological order?
    # Not strictly necessary if we re-index correctly, but good for determinism.
    sorted_needed_gates = sorted(needed_gates)
    
    for k in sorted_needed_gates:
        a, b = gates[k]
        new_a = old_to_new[a]
        new_b = old_to_new[b]
        new_k = len(new_gates)
        new_wire = n + 1 + new_k
        new_gates.append([new_a, new_b])
        old_to_new[n + 1 + k] = new_wire
    
    # Step 5: Update outputs
    final_outputs = []
    for wires in row_wires:
        if not wires:
            final_outputs.append(n)
        else:
            # XOR all wires in the list
            # We need to create the XOR chain if len(wires) > 1
            # But wait, the wires are already available as wires in the new circuit.
            # We just need to XOR them together.
            # However, the interface requires 'gates' to be the list of ALL gates.
            # The 'outputs' are just wire indices.
            # So if a row is XOR of w1, w2, w3, we need to create gates for w1^w2, then (w1^w2)^w3.
            # But those gates are NOT in the 'needed_gates' set from the CSE phase!
            # The CSE phase only created gates for the intermediate subexpressions.
            # The final XORs to combine them into the row output are NOT included.
            
            # This is a problem. The CSE creates shared intermediates, but we still need
            # to combine them to form the final output.
            # The cost of combining is the number of XORs in the row representation.
            
            # We need to include these final combination gates in the gate list.
            # Let's add them now.
            
            current = old_to_new[wires[0]]
            for w in wires[1:]:
                new_w = old_to_new[w]
                gate_idx = len(new_gates)
                new_wire = n + 1 + gate_idx
                new_gates.append([current, new_w])
                old_to_new[new_wire] = new_wire  # Not needed, but for consistency
                current = new_wire
            
            final_outputs.append(current)
    
    return {'gates': new_gates, 'outputs': final_outputs}

def synthesize(matrix, depth_limit):
    """
    Entry point. Returns a dictionary with 'gates' and 'outputs'.
    """
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    start_time = time.time()
    total_budget = 85.0
    
    # Start with baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_gates = baseline['gates'][:]
    best_outputs = baseline['outputs'][:]
    best_count = len(best_gates)
    
    if best_count == 0:
        return {'gates': best_gates, 'outputs': best_outputs}
    
    # Try CSE synthesis
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5.0:
        try:
            cse_result = _cse_synthesis(matrix, depth_limit, time_budget=min(remaining - 2.0, 50.0))
            if _verify_solution(matrix, cse_result['gates'], cse_result['outputs'], depth_limit):
                if len(cse_result['gates']) < best_count:
                    best_gates = cse_result['gates'][:]
                    best_outputs = cse_result['outputs'][:]
                    best_count = len(best_gates)
        except Exception:
            pass
    
    # Final safety check
    if not _verify_solution(matrix, best_gates, best_outputs, depth_limit):
        return {'gates': baseline['gates'], 'outputs': baseline['outputs']}
    
    return {'gates': best_gates, 'outputs': best_outputs}

# EVOLVE-BLOCK-END
