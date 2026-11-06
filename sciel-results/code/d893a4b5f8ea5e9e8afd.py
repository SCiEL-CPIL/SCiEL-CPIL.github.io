import time
import random
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline_circuit():
        """Independent XOR of inputs for each output."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _build_circuit_from_exprs(exprs, gates_so_far, next_wire):
        """Build the final circuit from expression bitsets and existing gates."""
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    # XOR all wires together
                    prev = wires[0]
                    for w in wires[1:]:
                        gates_so_far.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates_so_far, 'outputs': outputs}

    def _set_cover_synthesis(time_budget, seed):
        """
        Minimum-weight set cover formulation for XOR-circuit synthesis.
        
        Universe: required output vectors (unique non-zero rows)
        Candidate subsets: minimal linear dependencies (cycles) in the affine span
        Cost of a subset: number of new intermediate gates to resolve that dependency
        
        Greedy set cover: iteratively select the dependency that covers the most
        uncovered outputs per unit cost.
        """
        rng = random.Random(seed)
        
        # Get unique non-zero target vectors
        unique_targets = list(set(row_bits))
        unique_targets = [t for t in unique_targets if t != 0]
        
        if not unique_targets:
            return {'gates': [], 'outputs': [n] * m}
        
        # Initial known vectors: inputs and zero
        known_vecs = {}  # vec -> wire_id
        for i in range(n):
            known_vecs[1 << i] = i
        known_vecs[0] = n
        
        gates = []
        next_wire = n + 1
        
        # Track which targets are covered (have a wire)
        covered_targets = set()
        
        def _get_wire_for_vec(vec):
            """Get wire ID for a vector, or None if not known."""
            return known_vecs.get(vec)
        
        def _cost_to_produce_direct(vec):
            """Cost to produce vec directly from inputs."""
            pc = bin(vec).count('1')
            return max(0, pc - 1)
        
        def _find_minimal_dependencies(pending_targets, known_vecs_dict):
            """
            Find minimal linear dependencies among pending targets and known vectors.
            A dependency is a set of vectors {v1, v2, ..., vk} such that v1 ^ v2 ^ ... ^ vk = 0.
            We look for short cycles: pairs, triples, etc.
            """
            dependencies = []
            
            # Check for pairs: vi ^ vj = 0 means vi = vj (already covered)
            # Check for triples: vi ^ vj ^ vk = 0
            # More generally, find small sets that XOR to zero
            
            # For efficiency, only check dependencies involving at most 3-4 vectors
            # that include at least one pending target
            
            pending_list = list(pending_targets)
            
            # Pair dependencies: if vi ^ vj is in known_vecs, we can produce vi from vj and known
            # This is already handled by the cost function
            
            # Triple dependencies: vi ^ vj ^ vk = 0
            # If two of them are known, the third can be produced with 1 gate
            # If one is known and two are pending, we can produce both with 2 gates (saving 1)
            
            # Let's find all minimal dependencies of size 3 or 4
            for i_idx in range(len(pending_list)):
                vi = pending_list[i_idx]
                
                # Check if vi can be made from two known vectors (already in cost)
                # Check dependencies with other pending targets
                
                for j_idx in range(i_idx + 1, len(pending_list)):
                    vj = pending_list[j_idx]
                    xor_ij = vi ^ vj
                    
                    # If xor_ij is known, we have a dependency: vi ^ vj ^ xor_ij = 0
                    # Cost: 1 gate to produce xor_ij (if not already known)
                    # But xor_ij is known, so this means vi = vj ^ xor_ij
                    # This is already captured by the direct cost
                    
                    # Check if there's a third vector vk such that vi ^ vj ^ vk = 0
                    # i.e., vk = vi ^ vj
                    vk = xor_ij
                    if vk in known_vecs:
                        # Dependency: {vi, vj, vk} with vk known
                        # Cost to resolve: 1 gate (produce vi from vj and vk, or vj from vi and vk)
                        # Benefit: covers 1 pending target with 1 gate instead of direct cost
                        dependencies.append({
                            'vectors': [vi, vj, vk],
                            'pending_in_dep': [vi, vj],
                            'known_in_dep': [vk],
                            'cost': 1,
                            'type': 'triple_with_known'
                        })
                    
                    # Check if vk is also pending
                    if vk in pending_targets and vk not in (vi, vj):
                        # Dependency: {vi, vj, vk} all pending
                        # Cost to resolve: 2 gates (produce one from the other two)
                        # Benefit: covers 2 pending targets with 2 gates
                        dependencies.append({
                            'vectors': [vi, vj, vk],
                            'pending_in_dep': [vi, vj, vk],
                            'known_in_dep': [],
                            'cost': 2,
                            'type': 'triple_all_pending'
                        })
                
                # Check 4-vector dependencies: vi ^ vj ^ vk ^ vl = 0
                # This is more expensive to check, so limit it
                if len(pending_list) <= 20:  # Only do this for small instances
                    for k_idx in range(j_idx + 1, len(pending_list)):
                        vk = pending_list[k_idx]
                        xor_ijk = vi ^ vj ^ vk
                        
                        # Check if xor_ijk is known
                        if xor_ijk in known_vecs:
                            # Dependency: {vi, vj, vk, xor_ijk} with xor_ijk known
                            # Cost: 2 gates (produce vi from vj^vk^xor_ijk, etc.)
                            # This is getting complex; skip for now
                            pass
                        
                        # Check if xor_ijk is pending
                        if xor_ijk in pending_targets and xor_ijk not in (vi, vj, vk):
                            # Dependency: {vi, vj, vk, xor_ijk} all pending
                            # Cost: 3 gates
                            # Only consider if it saves gates vs direct
                            direct_cost = sum(_cost_to_produce_direct(v) for v in [vi, vj, vk, xor_ijk])
                            if 3 < direct_cost:
                                dependencies.append({
                                    'vectors': [vi, vj, vk, xor_ijk],
                                    'pending_in_dep': [vi, vj, vk, xor_ijk],
                                    'known_in_dep': [],
                                    'cost': 3,
                                    'type': 'quad_all_pending'
                                })
            
            return dependencies
        
        def _resolve_dependency(dep, known_vecs_dict, gates_list, next_wire_ref):
            """
            Resolve a dependency by adding gates.
            Returns (new_gates, new_known_vecs, new_next_wire)
            """
            new_gates = []
            new_known = dict(known_vecs_dict)
            new_next_wire = next_wire_ref
            
            vectors = dep['vectors']
            pending_in_dep = dep['pending_in_dep']
            known_in_dep = dep['known_in_dep']
            
            if dep['type'] == 'triple_with_known':
                # vi ^ vj ^ vk = 0, vk is known
                # Produce vi from vj and vk: vi = vj ^ vk
                # But vj is also pending, so we need to produce vj first or use another approach
                # Actually, if vk is known, we can produce either vi or vj with 1 gate
                # Let's produce the one with higher direct cost
                vi, vj, vk = vectors[0], vectors[1], vectors[2]
                cost_vi_direct = _cost_to_produce_direct(vi)
                cost_vj_direct = _cost_to_produce_direct(vj)
                
                if cost_vi_direct >= cost_vj_direct:
                    # Produce vi from vj and vk
                    # But vj is pending! We need vj first.
                    # This dependency doesn't directly help unless vj is also produced
                    # Let's just produce vi using vk and the inputs that make up vj
                    # Actually, this is getting too complex. Let's simplify:
                    # If vk is known, vi = vj ^ vk. If we can produce vj cheaply, this helps.
                    # For now, just add the gate vi = vj ^ vk and mark vi as produced
                    # But vj must be available...
                    
                    # Alternative: produce vi directly, and this dependency tells us
                    # that vj = vi ^ vk, so if we produce vi, vj comes for free
                    # Let's produce the one with lower direct cost
                    if cost_vi_direct <= cost_vj_direct:
                        # Produce vi directly
                        idxs = [j for j in range(n) if (vi >> j) & 1]
                        if len(idxs) == 1:
                            wire_vi = idxs[0]
                        else:
                            prev = idxs[0]
                            for idx in idxs[1:]:
                                new_gates.append([prev, idx])
                                prev = new_next_wire
                                new_next_wire += 1
                            wire_vi = prev
                        new_known[vi] = wire_vi
                        # Now vj = vi ^ vk
                        wire_vj = new_next_wire
                        new_gates.append([wire_vi, new_known[vk]])
                        new_next_wire += 1
                        new_known[vj] = wire_vj
                    else:
                        # Produce vj directly
                        idxs = [j for j in range(n) if (vj >> j) & 1]
                        if len(idxs) == 1:
                            wire_vj = idxs[0]
                        else:
                            prev = idxs[0]
                            for idx in idxs[1:]:
                                new_gates.append([prev, idx])
                                prev = new_next_wire
                                new_next_wire += 1
                            wire_vj = prev
                        new_known[vj] = wire_vj
                        # Now vi = vj ^ vk
                        wire_vi = new_next_wire
                        new_gates.append([wire_vj, new_known[vk]])
                        new_next_wire += 1
                        new_known[vi] = wire_vi
                else:
                    # Symmetric case
                    if cost_vj_direct <= cost_vi_direct:
                        # Produce vj directly
                        idxs = [j for j in range(n) if (vj >> j) & 1]
                        if len(idxs) == 1:
                            wire_vj = idxs[0]
                        else:
                            prev = idxs[0]
                            for idx in idxs[1:]:
                                new_gates.append([prev, idx])
                                prev = new_next_wire
                                new_next_wire += 1
                            wire_vj = prev
                        new_known[vj] = wire_vj
                        # Now vi = vj ^ vk
                        wire_vi = new_next_wire
                        new_gates.append([wire_vj, new_known[vk]])
                        new_next_wire += 1
                        new_known[vi] = wire_vi
                    else:
                        # Produce vi directly
                        idxs = [j for j in range(n) if (vi >> j) & 1]
                        if len(idxs) == 1:
                            wire_vi = idxs[0]
                        else:
                            prev = idxs[0]
                            for idx in idxs[1:]:
                                new_gates.append([prev, idx])
                                prev = new_next_wire
                                new_next_wire += 1
                            wire_vi = prev
                        new_known[vi] = wire_vi
                        # Now vj = vi ^ vk
                        wire_vj = new_next_wire
                        new_gates.append([wire_vi, new_known[vk]])
                        new_next_wire += 1
                        new_known[vj] = wire_vj
            
            elif dep['type'] == 'triple_all_pending':
                # vi ^ vj ^ vk = 0, all pending
                # Produce vi and vj directly, then vk = vi ^ vj with 1 gate
                # Or produce the two with lowest direct costs, then the third
                vi, vj, vk = vectors[0], vectors[1], vectors[2]
                costs = [_cost_to_produce_direct(v) for v in [vi, vj, vk]]
                vecs = [vi, vj, vk]
                
                # Sort by cost
                order = sorted(range(3), key=lambda i: costs[i])
                # Produce the two cheapest directly
                for idx in order[:2]:
                    v = vecs[idx]
                    idxs = [j for j in range(n) if (v >> j) & 1]
                    if len(idxs) == 1:
                        wire_v = idxs[0]
                    else:
                        prev = idxs[0]
                        for idx2 in idxs[1:]:
                            new_gates.append([prev, idx2])
                            prev = new_next_wire
                            new_next_wire += 1
                        wire_v = prev
                    new_known[v] = wire_v
                
                # Produce the third from the other two
                idx_third = order[2]
                v_third = vecs[idx_third]
                v_a = vecs[order[0]]
                v_b = vecs[order[1]]
                wire_third = new_next_wire
                new_gates.append([new_known[v_a], new_known[v_b]])
                new_next_wire += 1
                new_known[v_third] = wire_third
            
            elif dep['type'] == 'quad_all_pending':
                # vi ^ vj ^ vk ^ vl = 0, all pending
                # Produce three directly, then the fourth with 1 gate
                vi, vj, vk, vl = vectors[0], vectors[1], vectors[2], vectors[3]
                costs = [_cost_to_produce_direct(v) for v in [vi, vj, vk, vl]]
                vecs = [vi, vj, vk, vl]
                
                # Sort by cost
                order = sorted(range(4), key=lambda i: costs[i])
                # Produce the three cheapest directly
                for idx in order[:3]:
                    v = vecs[idx]
                    idxs = [j for j in range(n) if (v >> j) & 1]
                    if len(idxs) == 1:
                        wire_v = idxs[0]
                    else:
                        prev = idxs[0]
                        for idx2 in idxs[1:]:
                            new_gates.append([prev, idx2])
                            prev = new_next_wire
                            new_next_wire += 1
                        wire_v = prev
                    new_known[v] = wire_v
                
                # Produce the fourth from the other three: vl = vi ^ vj ^ vk
                # Need two gates: temp = vi ^ vj, then vl = temp ^ vk
                idx_fourth = order[3]
                v_fourth = vecs[idx_fourth]
                v_a = vecs[order[0]]
                v_b = vecs[order[1]]
                v_c = vecs[order[2]]
                
                wire_temp = new_next_wire
                new_gates.append([new_known[v_a], new_known[v_b]])
                new_next_wire += 1
                wire_fourth = new_next_wire
                new_gates.append([wire_temp, new_known[v_c]])
                new_next_wire += 1
                new_known[v_fourth] = wire_fourth
            
            return new_gates, new_known, new_next_wire
        
        # Greedy set cover loop
        max_iter = 300
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            
            pending = [t for t in unique_targets if t not in covered_targets]
            if not pending:
                break
            
            # Find dependencies
            deps = _find_minimal_dependencies(set(pending), known_vecs)
            
            if not deps:
                # No dependencies found; produce remaining targets directly
                for t in pending:
                    idxs = [j for j in range(n) if (t >> j) & 1]
                    if len(idxs) == 1:
                        wire_t = idxs[0]
                    else:
                        prev = idxs[0]
                        for idx in idxs[1:]:
                            gates.append([prev, idx])
                            prev = next_wire
                            next_wire += 1
                        wire_t = prev
                    known_vecs[t] = wire_t
                    covered_targets.add(t)
                break
            
            # Score each dependency: benefit / cost
            # Benefit = number of pending targets covered
            # Use a smooth surrogate with randomization
            best_dep = None
            best_score = -1
            
            rng.shuffle(deps)
            for dep in deps:
                benefit = len(dep['pending_in_dep'])
                cost = dep['cost']
                # Smooth score: benefit / (cost + epsilon) with noise
                score = benefit / (cost + 0.1) + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_dep = dep
            
            if best_dep is None:
                break
            
            # Resolve the dependency
            new_gates, new_known, new_next_wire = _resolve_dependency(
                best_dep, known_vecs, gates, next_wire
            )
            gates.extend(new_gates)
            known_vecs = new_known
            next_wire = new_next_wire
            
            # Mark covered targets
            for t in best_dep['pending_in_dep']:
                if t in known_vecs:
                    covered_targets.add(t)
        
        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in known_vecs:
                outputs.append(known_vecs[e])
            else:
                # Fallback: direct
                idxs = [j for j in range(n) if (e >> j) & 1]
                if len(idxs) == 1:
                    outputs.append(idxs[0])
                else:
                    prev = idxs[0]
                    for idx in idxs[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}

    def _greedy_cse(time_budget, seed):
        """
        Greedy Common Subexpression Elimination with deterministic pair selection
        and bit-count tie-breaking to avoid random local optima.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1

        def _get_wires(e):
            """Get list of wire indices present in expression e."""
            wires = []
            temp = e
            w = 0
            while temp:
                if temp & 1:
                    wires.append(w)
                temp >>= 1
                w += 1
            return wires

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Count co-occurrence of each pair of wires
            co_occ = {}
            for e in exprs:
                if e == 0:
                    continue
                wires_in_e = _get_wires(e)

                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (min(a, b), max(a, b))
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            # Find the pair with highest count, using bit-count tie-breaking
            best_pair = None
            best_count = 1
            best_bit_count = float('inf')
            
            for (a, b), cnt in co_occ.items():
                if cnt > best_count or (cnt == best_count and (a + b) < best_bit_count):
                    best_count = cnt
                    best_bit_count = a + b
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        return _build_circuit_from_exprs(exprs, gates, next_wire)

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('set_cover', _set_cover_synthesis),
        ('greedy_cse', _greedy_cse),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    for strategy_name, strategy_func in strategies:
        for seed in seeds:
            if time.time() > deadline - 3.0:
                break

            try:
                result = strategy_func(deadline - 3.0, seed)
                if _verify(result):
                    d = _compute_depth(result['gates'], result['outputs'])
                    if d <= depth_limit:
                        gc = len(result['gates'])
                        if gc < best_gate_count:
                            best_gate_count = gc
                            best_result = result
            except Exception:
                pass

    return best_result
