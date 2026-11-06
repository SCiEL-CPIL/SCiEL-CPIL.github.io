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

    def _greedy_cse(time_budget, seed):
        """
        Improved Greedy Common Subexpression Elimination using a priority queue
        with a smooth surrogate gain function. Instead of just counting co-occurrences,
        we estimate the total gate reduction from introducing a new intermediate wire.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))  # inputs 0..n-1 and zero wire n

        def _popcount(x):
            return bin(x).count('1')

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

        def _estimate_gain(a, b, exprs_list):
            """
            Estimate the gate-count reduction from introducing g = a XOR b.
            Uses a smooth surrogate: for each expression containing both a and b,
            the gain is 1 (one fewer XOR in the final fanout). Additionally,
            if g appears in multiple expressions, there's a fixed cost of 1 gate
            but potential savings in the final XOR chains.
            """
            a_bit = 1 << a
            b_bit = 1 << b
            gain = 0
            for e in exprs_list:
                if (e & a_bit) and (e & b_bit):
                    # This expression would use g instead of a and b
                    # In the final circuit, this saves 1 gate in the XOR chain
                    gain += 1
            # Net gain = (number of expressions that benefit) - 1 (cost of new gate)
            # But we also need to consider that the final XOR chains might be shorter
            return gain

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Count co-occurrence of each pair of wires and estimate gain
            co_occ = {}
            for e in exprs:
                if e == 0:
                    continue
                wires_in_e = _get_wires(e)

                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (a, b)
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            # Find the pair with highest estimated gain
            best_pair = None
            best_gain = 1  # Must save at least 1 gate to be worthwhile
            items = list(co_occ.items())
            rng.shuffle(items)
            for (a, b), cnt in items:
                # Use a smooth surrogate: gain = cnt - 1 (net saving)
                # Add a small random perturbation to break ties smoothly
                gain = cnt - 1 + rng.random() * 0.1
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        return _build_circuit_from_exprs(exprs, gates, next_wire)

    def _steiner_mst(time_budget, seed):
        """
        Improved Steiner tree approach with lookahead: when selecting the next
        vector to add, consider not just the immediate cost but also how well
        it helps produce remaining targets. Uses a soft-min surrogate to balance
        immediate cost against future benefit.
        """
        rng = random.Random(seed)

        wire_to_vec = {}
        vec_to_wire = {}

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        next_wire = n + 1
        gates = []

        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        pending = list(targets)
        pending_set = set(pending)

        def _cost_to_produce(t, known_vecs):
            """Compute cost to produce t from known vectors."""
            if t in known_vecs:
                return 0
            # Check if t = a XOR b for any a, b in known
            for a_vec in known_vecs:
                b_vec = t ^ a_vec
                if b_vec in known_vecs:
                    return 1
            # Direct cost
            pc = bin(t).count('1')
            return max(0, pc - 1)

        def _future_benefit(t, pending_list, known_vecs):
            """Estimate how many pending targets become cheaper if we add t."""
            benefit = 0
            new_known = known_vecs | {t}
            for p in pending_list:
                old_cost = _cost_to_produce(p, known_vecs)
                new_cost = _cost_to_produce(p, new_known)
                if new_cost < old_cost:
                    benefit += (old_cost - new_cost)
            return benefit

        max_iter = 400
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending:
                break

            best_target = None
            best_score = float('inf')
            best_indices = None

            # Shuffle pending for randomization
            rng.shuffle(pending)

            known_vecs = set(vec_to_wire.keys())

            for t in pending:
                if t in vec_to_wire:
                    best_target = t
                    best_score = 0
                    best_indices = None
                    break

                # Try to compute t from two existing vectors
                found = False
                best_pair_for_t = None
                for w_vec, w_wire in wire_to_vec.items():
                    u_vec = t ^ w_vec
                    if u_vec in vec_to_wire:
                        u_wire = vec_to_wire[u_vec]
                        best_pair_for_t = (u_wire, w_wire)
                        found = True
                        break

                if found:
                    cost = 1
                else:
                    pc = bin(t).count('1')
                    cost = max(0, pc - 1)
                    best_pair_for_t = 'direct'

                # Use a smooth surrogate: score = cost - alpha * future_benefit
                # Alpha balances immediate cost vs. future savings
                alpha = 0.5
                benefit = _future_benefit(t, pending, known_vecs)
                score = cost - alpha * benefit + rng.random() * 0.05

                if score < best_score:
                    best_score = score
                    best_target = t
                    best_indices = best_pair_for_t

            if best_target is None:
                break

            t = best_target

            if t in vec_to_wire:
                pending.remove(t)
                pending_set.discard(t)
                continue

            if best_indices == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                u_wire, w_wire = best_indices
                gates.append([u_wire, w_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            pending.remove(t)
            pending_set.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    def _hypercube_mst(time_budget, seed):
        """
        Improved hypercube MST: Use a priority-queue-based Prim-like algorithm
        that considers intermediate vectors (not just targets) as potential
        additions to the tree. Uses a smooth surrogate to balance the immediate
        cost of adding a vector against its potential to reduce costs of other
        pending targets.
        """
        rng = random.Random(seed)

        # Known vectors: wire -> vector
        wire_to_vec = {}
        vec_to_wire = {}
        next_wire = n + 1
        gates = []

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        # Targets are the unique non-zero row vectors
        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        tree_vecs = set(vec_to_wire.keys())
        pending_targets = set(targets)

        def _cost_from_tree(cand, tree):
            """Cost to produce cand from vectors in tree."""
            if cand in tree:
                return 0, None
            for a_vec in tree:
                b_vec = cand ^ a_vec
                if b_vec in tree:
                    return 1, (vec_to_wire[a_vec], vec_to_wire[b_vec])
            pc = bin(cand).count('1')
            return max(0, pc - 1), 'direct'

        def _benefit_to_pending(cand, pending, tree):
            """How much does adding cand help with pending targets?"""
            benefit = 0
            new_tree = tree | {cand}
            for p in pending:
                if p in tree:
                    continue
                old_cost, _ = _cost_from_tree(p, tree)
                new_cost, _ = _cost_from_tree(p, new_tree)
                if new_cost < old_cost:
                    benefit += (old_cost - new_cost)
            return benefit

        max_iter = 600
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending_targets:
                break

            # Generate candidate vectors: targets and their XORs with tree vectors
            candidates = set()
            for t in pending_targets:
                candidates.add(t)
                # Limit the number of intermediate candidates for efficiency
                tree_list = list(tree_vecs)
                if len(tree_list) <= 50:
                    for w_vec in tree_list:
                        candidates.add(t ^ w_vec)
                else:
                    # Sample a subset
                    sample_size = min(20, len(tree_list))
                    sampled = rng.sample(tree_list, sample_size)
                    for w_vec in sampled:
                        candidates.add(t ^ w_vec)

            best_candidate = None
            best_score = float('inf')
            best_pair = None

            for cand in candidates:
                if cand in tree_vecs:
                    continue

                cost, pair = _cost_from_tree(cand, tree_vecs)

                # Only consider candidates that are targets or have high benefit
                if cand not in pending_targets:
                    benefit = _benefit_to_pending(cand, pending_targets, tree_vecs)
                    if benefit <= 0:
                        continue

                # Smooth surrogate: score = cost - beta * benefit + noise
                beta = 0.3
                if cand in pending_targets:
                    benefit = _benefit_to_pending(cand, pending_targets, tree_vecs)
                else:
                    benefit = _benefit_to_pending(cand, pending_targets, tree_vecs)

                score = cost - beta * benefit + rng.random() * 0.05

                if score < best_score:
                    best_score = score
                    best_candidate = cand
                    best_pair = pair

            if best_candidate is None:
                break

            t = best_candidate
            cost, pair = _cost_from_tree(t, tree_vecs)

            if pair == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                a_wire, b_wire = pair
                gates.append([a_wire, b_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            tree_vecs.add(t)
            pending_targets.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    def _basis_reduction(time_budget, seed):
        """
        Improved basis reduction: Find a reduced basis for the row space,
        synthesize basis vectors with shared subexpressions, then express
        each row as a combination of basis vectors.
        """
        rng = random.Random(seed)

        # Find a basis for the row space using Gaussian elimination
        # with row reduction to minimize popcounts
        basis = []
        row_to_basis_combo = [0] * m  # Each row as combination of basis vectors

        for i in range(m):
            v = row_bits[i]
            combo = 0
            if v == 0:
                row_to_basis_combo[i] = 0
                continue

            # Reduce v by existing basis vectors
            for j, b in enumerate(basis):
                if (v & b) and bin(v & b).count('1') <= bin(v).count('1'):
                    new_v = v ^ b
                    if bin(new_v).count('1') < bin(v).count('1'):
                        v = new_v
                        combo |= (1 << j)

            if v == 0:
                row_to_basis_combo[i] = combo
                continue

            # Add v to basis
            basis.append(v)
            row_to_basis_combo[i] = combo | (1 << len(basis) - 1)

        # Synthesize basis vectors with shared subexpressions
        # Use the greedy CSE approach on the basis vectors
        basis_exprs = list(basis)
        basis_gates = []
        basis_next_wire = n + 1
        basis_pool = list(range(n + 1))

        def _get_wires(e):
            wires = []
            temp = e
            w = 0
            while temp:
                if temp & 1:
                    wires.append(w)
                temp >>= 1
                w += 1
            return wires

        # Apply CSE to basis vectors
        max_cse_iter = 100
        for _ in range(max_cse_iter):
            if time.time() > time_budget:
                break

            co_occ = {}
            for e in basis_exprs:
                if e == 0:
                    continue
                wires_in_e = _get_wires(e)
                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (a, b)
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            best_pair = None
            best_count = 1
            items = list(co_occ.items())
            rng.shuffle(items)
            for (a, b), cnt in items:
                if cnt > best_count:
                    best_count = cnt
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = basis_next_wire
            basis_next_wire += 1
            basis_gates.append([a, b])
            basis_pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(len(basis_exprs)):
                e = basis_exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    basis_exprs[i] = e

        # Now build the full circuit
        gates = list(basis_gates)
        next_wire = basis_next_wire
        outputs = []

        # First, determine the wire for each basis vector
        basis_wires = []
        for bv in basis_exprs:
            if bv == 0:
                basis_wires.append(n)
            else:
                wires = _get_wires(bv)
                if len(wires) == 1:
                    basis_wires.append(wires[0])
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    basis_wires.append(prev)

        # Express each row as combination of basis vectors
        for i in range(m):
            combo = row_to_basis_combo[i]
            if combo == 0:
                outputs.append(n)
            else:
                # XOR together the basis vectors in the combination
                indices = [j for j in range(len(basis)) if (combo >> j) & 1]
                if len(indices) == 1:
                    outputs.append(basis_wires[indices[0]])
                else:
                    prev = basis_wires[indices[0]]
                    for idx in indices[1:]:
                        gates.append([prev, basis_wires[idx]])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)

        return {'gates': gates, 'outputs': outputs}

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('greedy_cse', _greedy_cse),
        ('steiner_mst', _steiner_mst),
        ('hypercube_mst', _hypercube_mst),
        ('basis_reduction', _basis_reduction),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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
