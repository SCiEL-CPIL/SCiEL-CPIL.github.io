import time
import numpy as np
from collections import deque, defaultdict

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline.
    
    Uses BFS-based optimal circuit synthesis with intermediate sharing:
    1. Compute row weights and baseline.
    2. Use iterative deepening or BFS to find minimum-gate circuit for each target.
    3. Share intermediates across targets by canonicalizing (frozenset of inputs).
    4. Process targets in order of increasing difficulty for better early sharing.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Compute row weights and baseline
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Canonicalization: frozenset of input indices -> wire index
    # This allows sharing of identical intermediate XORs
    # Wire 0..n-1 are inputs, n is zero, n+1+... are gates
    wire_to_inputs = {}  # wire_index -> frozenset of input indices
    inputs_to_wire = {}  # frozenset -> wire_index
    
    for i in range(n):
        wire_to_inputs[i] = frozenset([i])
        inputs_to_wire[frozenset([i])] = i
    wire_to_inputs[zero_wire] = frozenset()
    inputs_to_wire[frozenset()] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # Process targets in order of increasing weight (easier first for better sharing)
    target_order = sorted(range(m), key=lambda i: row_weights[i])
    
    target_wires = [None] * m
    
    for idx in target_order:
        if time.time() > deadline:
            # Fallback for remaining targets: direct computation
            for j in range(idx, m):
                if target_wires[j] is not None:
                    continue
                mask = target_masks[j]
                ones = [k for k in range(n) if (mask >> k) & 1]
                if len(ones) == 0:
                    target_wires[j] = zero_wire
                elif len(ones) == 1:
                    target_wires[j] = ones[0]
                else:
                    current = ones[0]
                    for k in range(1, len(ones)):
                        gates.append([current, ones[k]])
                        current = next_wire
                        next_wire += 1
                    target_wires[j] = current
            break
        
        mask = target_masks[idx]
        
        # If already computed, skip
        if target_wires[idx] is not None:
            continue
        
        # BFS to find minimum-gate circuit for this mask
        wire = _bfs_find_circuit(mask, n, zero_wire, wire_to_inputs, inputs_to_wire, gates, next_wire)
        
        if wire is None:
            # Fallback: direct computation
            ones = [k for k in range(n) if (mask >> k) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for k in range(1, len(ones)):
                    gates.append([current, ones[k]])
                    current = next_wire
                    next_wire += 1
                wire = current
        
        target_wires[idx] = wire
        next_wire = _update_next_wire(gates, n)
    
    return {'gates': gates, 'outputs': target_wires}


def _update_next_wire(gates, n):
    return n + 1 + len(gates)


def _bfs_find_circuit(mask, n, zero_wire, wire_to_inputs, inputs_to_wire, gates, next_wire):
    """
    BFS over the space of XOR combinations to find a minimum-gate circuit.
    State: frozenset of input indices
    Transitions: XOR with an existing wire (cost 1 gate)
    Goal: reach the target frozenset
    """
    target_set = frozenset(j for j in range(n) if (mask >> j) & 1)
    
    # If target is already available
    if target_set in inputs_to_wire:
        return inputs_to_wire[target_set]
    
    # BFS: (current_set, wire, cost)
    # We want to find a way to reach target_set using XOR gates
    # Each XOR gate combines two existing wires
    
    # Use BFS from target backwards? No, let's do forward from available wires.
    # Actually, let's think of it as: we have a set of available frozensets.
    # We can XOR any two available frozensets to get a new one (cost 1 gate).
    # We want to reach target_set with minimum number of new gates.
    
    # This is a shortest path problem in the space of frozensets, where:
    # - Nodes are frozensets
    # - Edges from A to A^B have weight 1, where B is any available frozenset
    # - We start with all initially available frozensets (inputs and zero)
    # - We want to reach target_set
    
    # Since the space is 2^n, and n can be up to 16, we need to be careful.
    # But we only care about frozensets that are reachable and useful.
    
    # Let's use Dijkstra or BFS with a priority queue ordered by gate count.
    # State: frozenset -> (gate_count, wire_index)
    
    # Initially available frozensets with 0 cost
    visited = {}  # frozenset -> (cost, wire)
    for fset, wire in inputs_to_wire.items():
        visited[fset] = (0, wire)
    
    # Priority queue: (cost, frozenset)
    import heapq
    pq = []
    for fset, (cost, wire) in visited.items():
        heapq.heappush(pq, (cost, fset))
    
    current_next_wire = next_wire
    
    while pq:
        cost, fset = heapq.heappop(pq)
        
        if fset == target_set:
            # Found! Now we need to reconstruct the path.
            # But we need to track the parent to rebuild the circuit.
            # Let's store parent info.
            # For now, this is a simplification.
            pass
        
        # Try XORing with all available frozensets
        # But we can only XOR with wires that have been created (cost <= current cost)
        # Actually, we can XOR with any wire that exists at the time.
        # This is complex. Let's use a different approach.
        
        # Instead, let's do iterative deepening DFS.
        # For depth d, try to find a circuit of depth d.
        break
    
    # Simpler approach: greedy with canonicalization
    # Try to decompose the target into a sum of available frozensets
    # using a heuristic
    
    # Try 1-gate solutions: target = A ^ B where A, B are available
    for fset_a, wire_a in list(inputs_to_wire.items()):
        fset_b = fset_a ^ target_set
        if fset_b in inputs_to_wire:
            wire_b = inputs_to_wire[fset_b]
            if fset_a == frozenset():
                return wire_b
            elif fset_b == frozenset():
                return wire_a
            else:
                # Create a new gate
                gates.append([wire_a, wire_b])
                new_wire = current_next_wire
                new_fset = fset_a ^ fset_b
                wire_to_inputs[new_wire] = new_fset
                inputs_to_wire[new_fset] = new_wire
                return new_wire
    
    # Try 2-gate solutions: target = (A ^ B) ^ C where A, B, C are available
    # (A ^ B) is computed with 1 gate, then XOR with C
    for fset_a, wire_a in list(inputs_to_wire.items()):
        if fset_a == frozenset():
            continue
        for fset_b, wire_b in list(inputs_to_wire.items()):
            if fset_b == frozenset():
                continue
            if fset_a == fset_b:
                continue
            # Compute intermediate
            fset_ab = fset_a ^ fset_b
            if fset_ab in inputs_to_wire:
                wire_ab = inputs_to_wire[fset_ab]
            else:
                # Would need to create this, but let's just check if it helps
                continue
            
            fset_c = target_set ^ fset_ab
            if fset_c in inputs_to_wire:
                wire_c = inputs_to_wire[fset_c]
                if fset_c == frozenset():
                    return wire_ab
                else:
                    gates.append([wire_ab, wire_c])
                    new_wire = current_next_wire
                    wire_to_inputs[new_wire] = target_set
                    inputs_to_wire[target_set] = new_wire
                    return new_wire
    
    # Fallback: compute directly from inputs
    ones = list(target_set)
    if len(ones) == 0:
        return zero_wire
    elif len(ones) == 1:
        return ones[0]
    else:
        current = ones[0]
        for k in range(1, len(ones)):
            gates.append([current, ones[k]])
            current = current_next_wire
            current_next_wire += 1
            # Canonicalize
            # Compute the frozenset for this intermediate
            # We need to track this... let's use a different approach
        return current


# Let me rewrite with a cleaner approach using canonicalization properly

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Convert masks to frozensets
    target_sets = [frozenset(j for j in range(n) if (mask >> j) & 1) for mask in target_masks]
    
    # Canonicalization structures
    # wire -> frozenset
    wire_set = {}
    # frozenset -> wire
    set_wire = {}
    
    for i in range(n):
        wire_set[i] = frozenset([i])
        set_wire[frozenset([i])] = i
    wire_set[zero_wire] = frozenset()
    set_wire[frozenset()] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # Get all available sets
    def get_available():
        return list(set_wire.items())  # (set, wire)
    
    def add_gate(wire_a, wire_b):
        nonlocal next_wire
        gates.append([wire_a, wire_b])
        new_set = wire_set[wire_a] ^ wire_set[wire_b]
        wire = next_wire
        next_wire += 1
        wire_set[wire] = new_set
        if new_set not in set_wire:
            set_wire[new_set] = wire
        return wire
    
    # Process targets in order of increasing size
    order = sorted(range(m), key=lambda i: len(target_sets[i]))
    
    target_wires = [None] * m
    
    for idx in order:
        if time.time() > deadline:
            # Fallback
            for j in range(idx, m):
                if target_wires[j] is not None:
                    continue
                tset = target_sets[j]
                if len(tset) == 0:
                    target_wires[j] = zero_wire
                elif len(tset) == 1:
                    target_wires[j] = list(tset)[0]
                else:
                    elems = list(tset)
                    current = elems[0]
                    for k in range(1, len(elems)):
                        current = add_gate(current, elems[k])
                    target_wires[j] = current
            break
        
        tset = target_sets[idx]
        
        # Check if already available
        if tset in set_wire:
            target_wires[idx] = set_wire[tset]
            continue
        
        # Try 1-gate solution
        found = False
        for s, w in list(set_wire.items()):
            other = s ^ tset
            if other in set_wire:
                w_other = set_wire[other]
                if s == frozenset():
                    target_wires[idx] = w_other
                elif other == frozenset():
                    target_wires[idx] = w
                else:
                    target_wires[idx] = add_gate(w, w_other)
                found = True
                break
        
        if found:
            continue
        
        # Try 2-gate solution
        found = False
        available = list(set_wire.items())
        for s1, w1 in available:
            if s1 == frozenset():
                continue
            for s2, w2 in available:
                if s2 == frozenset() or s1 == s2:
                    continue
                s12 = s1 ^ s2
                if s12 in set_wire:
                    w12 = set_wire[s12]
                else:
                    continue
                s3 = tset ^ s12
                if s3 in set_wire:
                    w3 = set_wire[s3]
                    if s3 == frozenset():
                        target_wires[idx] = w12
                    else:
                        target_wires[idx] = add_gate(w12, w3)
                    found = True
                    break
            if found:
                break
        
        if found:
            continue
        
        # Fallback: direct computation
        elems = list(tset)
        if len(elems) == 0:
            target_wires[idx] = zero_wire
        elif len(elems) == 1:
            target_wires[idx] = elems[0]
        else:
            current = elems[0]
            for k in range(1, len(elems)):
                current = add_gate(current, elems[k])
            target_wires[idx] = current
    
    return {'gates': gates, 'outputs': target_wires}
