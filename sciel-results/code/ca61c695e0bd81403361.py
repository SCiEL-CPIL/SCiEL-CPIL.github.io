import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m_val = n // 2

    # Precompute edge indices
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Precompute edge arrays for fast objective
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)

    # Precompute edge types: 0=top cycle, 1=bottom cycle, 2=matching
    edge_type = np.zeros(num_edges, dtype=np.int32)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            edge_type[i] = 0
        elif u >= m_val and v >= m_val:
            edge_type[i] = 1
        else:
            edge_type[i] = 2

    # Precompute circulant shift indices for top and bottom cycles
    # For top cycle: edge i connects i to (i+1)%m
    # For bottom cycle: edge i connects (m+i) to (m+(i+1)%m)
    top_edge_to_vertex = {}
    bottom_edge_to_vertex = {}
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            top_edge_to_vertex[i] = u  # edge from u to v=(u+1)%m
        elif u >= m_val and v >= m_val:
            bottom_edge_to_vertex[i] = u - m_val  # edge from (m+u') to (m+v')

    def build_S(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[eu, ev] = signs
        S[ev, eu] = signs
        return S

    def get_objective(signs):
        S = build_S(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(signs):
        return is_connected_fast(signs)

    # Precompute 4-cycles (side faces of prism)
    cycles_4 = []
    for i in range(m_val):
        v1 = i
        v2 = (i + 1) % m_val
        v3 = m_val + i
        v4 = m_val + (i + 1) % m_val
        e1 = edge_idx.get((v1, v2))
        e2 = edge_idx.get((v2, v4))
        e3 = edge_idx.get((v4, v3))
        e4 = edge_idx.get((v3, v1))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycles_4.append([e1, e2, e3, e4])

    # Also precompute top cycle and bottom cycle edge lists
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            top_cycle_edges.append(i)
        elif u >= m_val and v >= m_val:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)

    rng = np.random.default_rng(3)

    # ====== Fast connectivity check using cycle structure ======
    def is_connected_fast(signs):
        """
        For a prism graph, the 2-lift is connected iff the "twist" around
        each of the two m-cycles is consistent in a specific way.
        
        Key insight: The prism has two m-cycles (top and bottom) connected by matching.
        In the 2-lift, a sign of -1 on an edge swaps the two sides.
        
        The lift is connected iff going around the top cycle and bottom cycle
        produces a non-trivial permutation that connects the two sides.
        
        More precisely: assign each vertex a "side" label (0 or 1) in the lift.
        Start at vertex 0, side 0. Traverse the top cycle: each edge with sign -1
        flips the side. After going around the top cycle (m edges), if the product
        of signs is -1, we end up on side 1. Similarly for bottom cycle.
        
        The matching edges connect top[i] to bottom[i]. In the lift, a matching edge
        with sign +1 connects (top[i], s) to (bottom[i], s), and sign -1 connects
        (top[i], s) to (bottom[i], 1-s).
        
        The lift is connected iff the "winding number" around the prism is odd,
        i.e., the product of all signs around any face that wraps around is -1.
        
        Actually, let's use a simpler criterion: the lift is connected iff there
        exists a path from (0,0) to (0,1) in the lift. This happens iff the
        product of signs along some cycle in the base that goes from 0 back to 0
        while traversing edges in a way that creates a "twist" is -1.
        
        For the prism, the simplest check: compute the parity of the number of -1
        signs on the top cycle and the bottom cycle. The lift is connected iff
        (top_parity XOR bottom_parity) matches the matching pattern in a way that
        connects the two sheets.
        
        Let me use a more robust approach: BFS on the 2n vertices but optimized.
        """
        # Use union-find with path compression
        parent = list(range(2 * n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            u0, u1 = 2 * u, 2 * u + 1
            v0, v1 = 2 * v, 2 * v + 1
            if s == 1:
                union(u0, v0)
                union(u1, v1)
            else:
                union(u0, v1)
                union(u1, v0)
        
        root = find(0)
        # Check if all vertices are in the same component
        for i in range(1, 2 * n):
            if find(i) != root:
                return False
        return True

    best_signs = None
    best_obj = float('inf')

    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    def local_search(initial_signs, time_limit=None):
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline
        current_signs = initial_signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        while time.time() < time_limit:
            improved = False

            # 1. Single flips
            best_flip_idx = -1
            best_flip_obj = current_obj
            for i in range(num_edges):
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < best_flip_obj - 1e-12:
                        best_flip_obj = obj
                        best_flip_idx = i
                current_signs[i] = old

            if best_flip_idx != -1:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_obj = best_flip_obj
                update_best(current_signs, current_obj)
                improved = True
            else:
                # 2. 4-cycle flips (preserve connectivity)
                best_cycle_idx = -1
                best_cycle_obj = current_obj
                for c_idx, cyc in enumerate(cycles_4):
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < best_cycle_obj - 1e-12:
                        best_cycle_obj = obj
                        best_cycle_idx = c_idx
                    for e in cyc:
                        current_signs[e] = -current_signs[e]

                if best_cycle_idx != -1:
                    for e in cycles_4[best_cycle_idx]:
                        current_signs[e] = -current_signs[e]
                    current_obj = best_cycle_obj
                    update_best(current_signs, current_obj)
                    improved = True
                else:
                    # 3. Top/bottom cycle flips
                    found = False
                    for cyc in [top_cycle_edges, bottom_cycle_edges]:
                        if len(cyc) == 0:
                            continue
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-12:
                                current_obj = obj
                                update_best(current_signs, current_obj)
                                found = True
                                improved = True
                                break
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                    if found:
                        continue

                    # 4. Pair flips (systematic, limited)
                    best_pair = None
                    best_pair_obj = current_obj
                    for a in range(num_edges):
                        if time.time() > time_limit:
                            break
                        for b in range(a + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < best_pair_obj - 1e-12:
                                    best_pair_obj = obj
                                    best_pair = (a, b)
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]

                    if best_pair is not None:
                        a, b = best_pair
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        current_obj = best_pair_obj
                        update_best(current_signs, current_obj)
                        improved = True
                    else:
                        # 5. Random triple flips
                        found_triple = False
                        for _ in range(20):
                            if time.time() > time_limit:
                                break
                            idxs = rng.choice(num_edges, size=3, replace=False)
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-12:
                                    current_obj = obj
                                    update_best(current_signs, current_obj)
                                    found_triple = True
                                    improved = True
                                    break
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                        if not found_triple:
                            break

    def polish(signs):
        nonlocal best_signs, best_obj
        current_signs = signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            improved = False
            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 1.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old
            if not improved:
                # 4-cycle sweep
                for cyc in cycles_4:
                    if time.time() > deadline - 1.0:
                        break
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
            if not improved:
                # Pair sweep
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-15:
                                current_obj = obj
                                improved = True
                                break
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                    if improved:
                        break
            if not improved:
                # Triple flip sweep (limited)
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        if time.time() > deadline - 1.0:
                            break
                        for c in range(b + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-15:
                                    current_obj = obj
                                    improved = True
                                    break
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        update_best(current_signs, current_obj)

    # ====== NEW: Phase-space continuous optimization ======
    def phase_optimize():
        """
        Continuous optimization over phases of a complex unit vector.
        The prism has a circulant structure: top cycle, bottom cycle, matching.
        We parameterize the sign assignment via phases and minimize the spectral radius
        of the signed adjacency using gradient descent in the continuous relaxation.
        """
        # Build the signed adjacency as a function of a continuous variable.
        # We relax signs to be in [-1, 1] and minimize max|eigenvalue|.
        # Use a smooth approximation: objective = sum of exp(lambda * |eig|) for large lambda,
        # or just use the max eigenvalue directly with subgradient.
        # 
        # Alternative: use the fact that for a symmetric matrix, the spectral radius
        # equals the max of |v^T S v| over unit v. So we can use a min-max formulation:
        # min_{x in [-1,1]^E} max_{||v||=1} |v^T S(x) v|
        #
        # We'll use gradient descent on x with a penalty for the max eigenvalue.
        # To make it differentiable, use a soft max: f(x) = (1/t) * log(sum exp(t*|eig_i|))
        # and compute gradients via automatic differentiation on the eigenvalues.
        #
        # Actually, for symmetric matrices, d/dx (max |eig|) can be computed via the
        # corresponding eigenvector. Let's use a simpler approach:
        # 
        # We'll use scipy's minimize with a custom objective that computes the
        # spectral radius and uses finite-difference gradients (or analytic via eigenvectors).
        
        from scipy.optimize import minimize

        def objective(x):
            """x is a vector in [-1, 1]^num_edges, return spectral radius of S(x)."""
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev = np.linalg.eigvalsh(S)
            return float(np.max(np.abs(ev)))

        def gradient(x):
            """Compute gradient of spectral radius w.r.t. x using eigenvectors."""
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev, evec = np.linalg.eigh(S)
            # The spectral radius is max |ev_i|. Find which eigenvalue(s) achieve it.
            abs_ev = np.abs(ev)
            max_val = np.max(abs_ev)
            # For the eigenvalue with largest |value|, the gradient of |ev_k| w.r.t. S
            # is (evec_k * evec_k^T) * sign(ev_k)
            # And dS/dx_i = E_i + E_i^T where E_i has 1 at (eu[i], ev[i]) and (ev[i], eu[i])
            # So d(max|ev|)/dx_i = sum over k achieving max: sign(ev_k) * (evec_k[eu[i]]*evec_k[ev[i]] + evec_k[ev[i]]*evec_k[eu[i]])
            # For simplicity, use the single eigenvalue with largest |value|
            k = np.argmax(abs_ev)
            sign_k = np.sign(ev[k])
            v_k = evec[:, k]
            grad = np.zeros(num_edges, dtype=np.float64)
            grad = sign_k * 2.0 * (v_k[eu] * v_k[ev])
            return grad

        # Multiple restarts in continuous space
        best_x = None
        best_f = float('inf')

        # Initializations:
        # 1. All 1s
        # 2. All -1s
        # 3. Random
        # 4. Structured: alternate on cycles
        inits = [
            np.ones(num_edges),
            -np.ones(num_edges),
            rng.uniform(-1, 1, size=num_edges),
            rng.uniform(-1, 1, size=num_edges),
        ]
        # Structured: top cycle alternating
        x_struct = np.ones(num_edges)
        for i, (u, v) in enumerate(edges):
            if u < m_val and v < m_val:
                x_struct[i] = 1.0 if (u % 2 == 0) else -1.0
            elif u >= m_val and v >= m_val:
                x_struct[i] = 1.0 if ((u - m_val) % 2 == 0) else -1.0
        inits.append(x_struct)

        for x0 in inits:
            if time.time() > deadline - 15.0:
                break
            try:
                res = minimize(
                    objective, x0, jac=gradient,
                    method='L-BFGS-B',
                    bounds=[(-1, 1)] * num_edges,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                f = res.fun
                if f < best_f:
                    best_f = f
                    best_x = res.x.copy()
            except Exception:
                pass

        if best_x is not None:
            # Round to signs
            signs = np.where(best_x >= 0, 1, -1).astype(np.int8)
            return signs, best_f

        return None, float('inf')

    # ====== Exhaustive search for small m ======
    def exhaustive_search():
        """Exhaustive search over all 2^E sign assignments. Only feasible for small E."""
        if num_edges > 16:
            return None, float('inf')
        best = None
        best_f = float('inf')
        for bits in product([1, -1], repeat=num_edges):
            signs = np.array(bits, dtype=np.int8)
            if not is_connected(signs):
                continue
            f = get_objective(signs)
            if f < best_f:
                best_f = f
                best = signs.copy()
        return best, best_f

    # ====== Main search ======

    # 1. Exhaustive search for m=4 (num_edges = 3*4 = 12, 2^12 = 4096 - feasible)
    if m_val == 4:
        ex_signs, ex_f = exhaustive_search()
        if ex_signs is not None:
            update_best(ex_signs, ex_f)

    # 2. Phase-space optimization
    phase_signs, phase_f = phase_optimize()
    if phase_signs is not None:
        if is_connected(phase_signs):
            local_search(phase_signs, deadline - 20.0)
            # Also polish the phase result directly
            polish(phase_signs)

    # 3. Standard local search from multiple initializations
    # All 1s
    signs = np.ones(num_edges, dtype=np.int8)
    if is_connected(signs):
        local_search(signs, deadline - 20.0)

    # Alternating top
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 15.0)

    # Alternating bottom
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u >= m_val and v >= m_val:
            signs[i] = 1 if ((u - m_val) % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 10.0)

    # Random restarts
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 5.0)

    # Final polish
    if best_signs is not None:
        polish(best_signs)

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    # Verify final result is valid
    if not is_connected(best_signs):
        # Fallback: try all +1
        fallback = np.ones(num_edges, dtype=np.int8)
        if is_connected(fallback):
            best_signs = fallback
        else:
            # Try alternating patterns
            for pattern in range(4):
                test = np.ones(num_edges, dtype=np.int8)
                for i, (u, v) in enumerate(edges):
                    if (u // m_val) == (v // m_val):  # same cycle
                        if (u % 2) != (pattern % 2):
                            test[i] = -1
                if is_connected(test):
                    best_signs = test
                    break

    return best_signs
