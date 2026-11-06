"""Exact XOR-circuit synthesis: Steiner-tree over XOR hypercube with greedy Prim."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from heapq import heappush, heappop

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

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
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    return True


def _baseline_synthesis(matrix, depth_limit=16):
    """Independent XOR chains per row."""
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
                gates.append([a, b])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}


def _steiner_tree_synthesis(matrix, depth_limit=16, time_budget=30.0):
    """
    Steiner-tree formulation:
    - Terminals: all target row vectors + zero vector
    - Candidate nodes: all vectors in the span of {inputs} that are "cheap" to reach
    - Edge weight between u and v: Hamming weight of (u XOR v) / 2 + 1  (proxy for gate cost)
    
    Actually, the true cost model: to produce vector v from two existing vectors a, b (where v = a XOR b),
    we need 1 gate. The "distance" in terms of number of gates from inputs to a vector is its
    circuit depth/cost. 
    
    Better formulation: We build a graph where:
    - Nodes are vectors in GF(2)^n
    - An edge (u, v) exists if we can produce v = u XOR w for some existing w, costing 1 gate
    - We want a Steiner tree connecting all terminals (targets + zero)
    
    Since the full hypercube is too large, we restrict to a candidate set:
    1. All input vectors (e_i) and zero
    2. All target vectors
    3. All XORs of pairs of targets
    4. All XORs of a target with a single input
    5. Vectors with small Hamming weight (<= 3) that are in the span
    
    Then run a greedy Prim-like algorithm:
    - Start with the set of "free" nodes (inputs + zero, which have depth 0)
    - Repeatedly add the cheapest edge that connects a new useful node to the tree
    - A node is "useful" if it helps produce a terminal or is a terminal
    
    For each added node v = a XOR b (where a, b are already in the tree), we pay 1 gate.
    The total gate count is the number of non-free nodes added.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Build candidate pool
    # Use bytes as keys for vectors
    pool_bytes = set()
    pool_vecs = []  # parallel list
    
    def add_vec(v):
        b = v.tobytes()
        if b not in pool_bytes:
            pool_bytes.add(b)
            pool_vecs.append(v)
            return True
        return False
    
    # Free nodes: inputs and zero
    free_vecs = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        add_vec(v)
        free_vecs.append(v)
    zero = np.zeros(n, dtype=np.int64)
    add_vec(zero)
    free_vecs.append(zero)
    
    # Targets
    targets = [matrix[i].copy() for i in range(m)]
    for t in targets:
        add_vec(t)
    
    # XOR of target pairs
    for i in range(m):
        for j in range(i + 1, m):
            add_vec((targets[i] + targets[j]) % 2)
    
    # XOR of target with each input
    for i in range(m):
        for j in range(n):
            v = (targets[i] + free_vecs[j]) % 2
            add_vec(v)
    
    # Small-weight vectors in the span: XOR of pairs of pool vectors with small weight
    pool_size = len(pool_vecs)
    for i in range(pool_size):
        w_i = np.count_nonzero(pool_vecs[i])
        if w_i > 3:
            continue
        for j in range(i + 1, pool_size):
            v = (pool_vecs[i] + pool_vecs[j]) % 2
            if np.count_nonzero(v) <= 3:
                add_vec(v)
    
    # Also add XOR of target with target-pair results (depth 2 intermediates)
    pool_size2 = len(pool_vecs)
    for i in range(m):
        for j in range(i + 1, m):
            tp = (targets[i] + targets[j]) % 2
            for k in range(n):
                add_vec((tp + free_vecs[k]) % 2)
    
    # Now run greedy Steiner tree
    # Node index mapping
    byte_to_idx = {v.tobytes(): i for i, v in enumerate(pool_vecs)}
    
    num_nodes = len(pool_vecs)
    is_free = [False] * num_nodes
    for i in range(len(free_vecs)):
        idx = byte_to_idx.get(free_vecs[i].tobytes(), -1)
        if idx >= 0:
            is_free[idx] = True
    
    # Which nodes are terminals?
    is_terminal = [False] * num_nodes
    for t in targets:
        idx = byte_to_idx.get(t.tobytes(), -1)
        if idx >= 0:
            is_terminal[idx] = True
    # Zero is also a terminal (for empty rows)
    zero_idx = byte_to_idx.get(zero.tobytes(), -1)
    if zero_idx >= 0:
        is_terminal[zero_idx] = True
    
    # Greedy Prim: maintain a "tree" set. Start with all free nodes.
    # For each non-tree node, track the cheapest way to connect it (via which two tree nodes).
    # Edge cost to add node v via (a, b) is 1 if v = pool[a] XOR pool[b] and a, b in tree.
    
    in_tree = [is_free[i] for i in range(num_nodes)]
    # parent info: for each node not in tree, best (cost, a_idx, b_idx) to connect
    best_cost = [float('inf')] * num_nodes
    best_a = [-1] * num_nodes
    best_b = [-1] * num_nodes
    
    # Initialize: for each non-tree node, check if it can be formed by XOR of two tree nodes
    # This is O(num_nodes * |tree|^2) which is expensive. Use a different approach.
    
    # Better: maintain for each pair of tree nodes, the XOR result.
    # tree_set = set of indices in tree
    tree_set = set(i for i in range(num_nodes) if is_free[i])
    
    # For each pair (a, b) in tree_set, compute v = pool[a] XOR pool[b], if v in pool and not in tree,
    # update best_cost[v] = 1, best_a[v] = a, best_b[v] = b
    
    # This is O(|tree|^2) per iteration. To speed up, we process in batches.
    
    # Use a priority queue of candidate edges (cost, node_idx, a_idx, b_idx)
    pq = []
    
    # Initial expansion: for each pair in tree_set
    tree_list = sorted(tree_set)
    for ii in range(len(tree_list)):
        a = tree_list[ii]
        va = pool_vecs[a]
        for jj in range(ii + 1, len(tree_list)):
            b = tree_list[jj]
            v = (va + pool_vecs[b]) % 2
            vb = v.tobytes()
            k = byte_to_idx.get(vb, -1)
            if k >= 0 and k not in tree_set:
                if best_cost[k] > 1:
                    best_cost[k] = 1
                    best_a[k] = a
                    best_b[k] = b
                    heappush(pq, (1, k, a, b))
    
    # Also, free nodes can be used as one side. Zero XOR v = v, so any target that is in pool
    # and equals pool[a] XOR zero for a in tree is free. Already handled above since zero is in tree.
    
    # Now greedily add nodes
    added_count = 0
    max_add = 500  # safety
    
    while added_count < max_add:
        if time.time() > deadline:
            break
        
        # Pop best edge
        found = False
        while pq:
            cost, k, a, b = heappop(pq)
            if k in tree_set:
                continue
            if cost != best_cost[k]:
                continue
            # Check a and b are still in tree
            if a not in tree_set or b not in tree_set:
                continue
            # Add k to tree
            tree_set.add(k)
            in_tree[k] = True
            added_count += 1
            found = True
            
            # Check if all terminals are in tree
            all_terminals_in = all(is_terminal[t] for t in range(num_nodes) if is_terminal[t])
            if all_terminals_in:
                break
            
            # Expand: for each node j in tree_set (newly added k with all existing),
            # compute pool[k] XOR pool[j]
            vk = pool_vecs[k]
            for j in tree_set:
                if j == k:
                    continue
                v = (vk + pool_vecs[j]) % 2
                vb = v.tobytes()
                nk = byte_to_idx.get(vb, -1)
                if nk >= 0 and nk not in tree_set:
                    if best_cost[nk] > 1:
                        best_cost[nk] = 1
                        best_a[nk] = k
                        best_b[nk] = j
                        heappush(pq, (1, nk, k, j))
                    elif best_cost[nk] == 1 and (best_a[nk], best_b[nk]) != (k, j):
                        # Alternative path, keep the first one
                        pass
            break
        
        if not found:
            break
        if all(is_terminal[t] for t in range(num_nodes) if is_terminal[t]):
            break
    
    # Check if all terminals are reachable
    missing_terminals = [t for t in range(num_nodes) if is_terminal[t] and not in_tree[t]]
    if missing_terminals:
        # Fallback: return None to signal failure
        return None
    
    # Now extract the circuit from the Steiner tree
    # Each non-free node k in tree was added via (best_a[k], best_b[k])
    # Build the gate list in topological order
    
    # Topological order: sort by "addition order" or by depth
    # Compute depth for each node
    node_depth = [0] * num_nodes
    for i in range(num_nodes):
        if is_free[i]:
            node_depth[i] = 0
    
    # We need to process in order of addition. Let's track addition order.
    # Actually, let's just do a topological sort based on parent pointers.
    
    # Build parent map
    parent_map = {}
    for k in range(num_nodes):
        if in_tree[k] and not is_free[k]:
            if best_a[k] >= 0 and best_b[k] >= 0:
                parent_map[k] = (best_a[k], best_b[k])
    
    # Topological sort: nodes with no parents first (free nodes), then their children
    # Use Kahn's algorithm
    in_degree = [0] * num_nodes
    children = [[] for _ in range(num_nodes)]
    for k, (a, b) in parent_map.items():
        in_degree[k] = 1
        children[a].append(k)
        children[b].append(k)
    
    from collections import deque
    dq = deque()
    for i in range(num_nodes):
        if in_degree[i] == 0 and in_tree[i]:
            dq.append(i)
    
    topo_order = []
    while dq:
        node = dq.popleft()
        topo_order.append(node)
        for child in children[node]:
            if in_tree[child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    dq.append(child)
    
    # Assign wire numbers
    pool_to_wire = {}
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        idx = byte_to_idx.get(v.tobytes(), -1)
        if idx >= 0:
            pool_to_wire[idx] = i
    zero_idx = byte_to_idx.get(zero.tobytes(), -1)
    if zero_idx >= 0:
        pool_to_wire[zero_idx] = n
    
    gates = []
    gate_counter = n + 1
    
    for idx in topo_order:
        if idx in pool_to_wire:
            continue
        if idx not in parent_map:
            continue
        a, b = parent_map[idx]
        wa = pool_to_wire.get(a)
        wb = pool_to_wire.get(b)
        if wa is None or wb is None:
            continue
        gates.append([wa, wb])
        pool_to_wire[idx] = gate_counter
        gate_counter += 1
    
    # Determine outputs
    outputs = []
    for i in range(m):
        t = targets[i]
        tb = t.tobytes()
        idx = byte_to_idx.get(tb, -1)
        if idx >= 0 and idx in pool_to_wire:
            outputs.append(pool_to_wire[idx])
        else:
            # Fallback for this row
            positions = np.where(t)[0].tolist()
            if len(positions) == 0:
                outputs.append(n)
            elif len(positions) == 1:
                outputs.append(positions[0])
            else:
                current = positions[0]
                for j in range(1, len(positions)):
                    gates.append([current, positions[j]])
                    current = n + 1 + len(gates) - 1
                outputs.append(current)
    
    # Verify depth
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    depth_arr = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        depth_arr[n + 1 + k] = 1 + max(depth_arr[a], depth_arr[b])
    max_depth = max(depth_arr[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
    for wire in outputs:
        max_depth = max(max_depth, depth_arr[wire])
    
    if max_depth > depth_limit:
        return None
    
    return {'gates': gates, 'outputs': outputs}


def _cse_polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """Common subexpression elimination polish."""
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break
        
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w
        
        improved = False
        
        # Remove unused gates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for ow in outputs:
                if ow == wire_k:
                    used = True
                    break
            if not used:
                for g in gates:
                    if g[0] == wire_k or g[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break
        
        if improved:
            continue
        
        # Redirect duplicates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k_bytes = wire_vec[wire_k].tobytes()
            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                gates.pop(k)
                improved = True
                break
        
        if not improved:
            break
    
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return {'gates': gates, 'outputs': outputs}


def _gaussian_elimination_synthesis(matrix, depth_limit=16, time_budget=10.0):
    """Gaussian elimination based synthesis."""
    m, n = matrix.shape
    rows = [matrix[i].copy() for i in range(m)]
    
    basis = []
    pivot_cols = []
    
    for row in rows:
        r = row.copy()
        for b, pc in zip(basis, pivot_cols):
            if r[pc] == 1:
                r = (r + b) % 2
        found = False
        for j in range(n):
            if r[j] == 1:
                for i, (b, pc) in enumerate(zip(basis, pivot_cols)):
                    if b[j] == 1:
                        basis[i] = (basis[i] + r) % 2
                basis.append(r)
                pivot_cols.append(j)
                found = True
                break
        if not found:
            continue
    
    rank = len(basis)
    if rank == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    gates = []
    basis_wires = []
    
    for b in basis:
        positions = np.where(b)[0].tolist()
        if len(positions) == 0:
            basis_wires.append(n)
        elif len(positions) == 1:
            basis_wires.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                gates.append([current, positions[j]])
                current = n + 1 + len(gates) - 1
            basis_wires.append(current)
    
    outputs = []
    for i in range(m):
        row = matrix[i].copy()
        coeffs = []
        for j, (b, pc) in enumerate(zip(basis, pivot_cols)):
            if row[pc] == 1:
                coeffs.append(j)
                row = (row + b) % 2
        
        if len(coeffs) == 0:
            outputs.append(n)
        elif len(coeffs) == 1:
            outputs.append(basis_wires[coeffs[0]])
        else:
            current = basis_wires[coeffs[0]]
            for j in range(1, len(coeffs)):
                gates.append([current, basis_wires[coeffs[j]]])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    depth_arr = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        depth_arr[n + 1 + k] = 1 + max(depth_arr[a], depth_arr[b])
    max_depth = max(depth_arr[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
    for wire in outputs:
        max_depth = max(max_depth, depth_arr[wire])
    
    if max_depth > depth_limit:
        return _baseline_synthesis(matrix, depth_limit)
    
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """Entry point."""
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    start_time = time.time()
    
    # Baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_result = baseline
    best_count = len(baseline['gates'])
    
    if best_count == 0:
        return baseline
    
    # Strategy 1: Steiner tree
    try:
        st_result = _steiner_tree_synthesis(matrix, depth_limit, time_budget=25.0)
        if st_result is not None:
            if len(st_result['gates']) < best_count:
                best_result = st_result
                best_count = len(st_result['gates'])
    except Exception:
        pass
    
    # Strategy 2: Gaussian elimination
    try:
        ge_result = _gaussian_elimination_synthesis(matrix, depth_limit, time_budget=10.0)
        if len(ge_result['gates']) < best_count:
            best_result = ge_result
            best_count = len(ge_result['gates'])
    except Exception:
        pass
    
    # CSE polish on best result
    elapsed = time.time() - start_time
    if elapsed < 50.0 and best_count > 0:
        try:
            polished = _cse_polish(matrix, best_result['gates'], best_result['outputs'],
                                   depth_limit, time_budget=10.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass
    
    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result
    
    # If time remains, try CSE on other results
    remaining = 55.0 - elapsed
    if remaining > 5.0:
        try:
            if len(baseline['gates']) < best_count:
                polished_base = _cse_polish(matrix, baseline['gates'], baseline['outputs'],
                                             depth_limit, time_budget=remaining - 2.0)
                if len(polished_base['gates']) < best_count:
                    best_result = polished_base
                    best_count = len(polished_base['gates'])
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
