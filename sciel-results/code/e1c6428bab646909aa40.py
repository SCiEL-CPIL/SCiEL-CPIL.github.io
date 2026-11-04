import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0  # safety margin before 150s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Build conflict graph: edge between u,v if Hamming distance <= 2
    # A covering code is an independent set in this graph
    # (no two codewords can be within distance 2 of each other... wait, that's not right)
    # Actually: a covering code requires every word within distance 1 of some codeword.
    # Two codewords CAN be close to each other. The constraint is just coverage.
    # So MIS framing doesn't directly apply. Let me rethink.
    #
    # Actually the problem is: minimize |C| such that union of balls radius 1 around C covers all space.
    # This is a set cover / dominating set problem on the Hamming graph.
    # The dual view: we want a minimum dominating set.
    #
    # Let me use a different approach: branch-and-bound on the dominating set problem
    # using bitset operations for speed.
    #
    # Alternative: Use the fact that for q=3, n=7, the space is 2187.
    # We can use a clever combination of:
    # 1. Greedy construction to get an initial upper bound
    # 2. Systematic removal with smart ordering (try removing codewords whose
    #    coverage is most redundant)
    # 3. Multiple random restarts with different greedy orderings
    # 4. Local search: swap operations (replace one codeword with another)
    
    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 10.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            # Add small random perturbation to break ties
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, max_time=10.0):
        """Add codewords to cover all uncovered points."""
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < max_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    # Phase 1: Get a good initial solution via multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(10):
        if (time.time() - t_start) > deadline - 30.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 25.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Iterative improvement - systematic removal with smart ordering
    # Key insight: remove codewords in order of lowest "criticality"
    # A codeword is critical if removing it would leave some point uncovered.
    # We want to remove non-critical codewords first.
    
    cur_code = best_code[:]
    cur_cov = build_cov(cur_code)
    cur_size = len(cur_code)

    # Build a list of codewords sorted by "removability"
    # A codeword c is removable if min(cur_cov[neighbors[c]]) >= 2
    # Priority: remove those with the most redundant coverage first
    
    improved = True
    while improved and (time.time() - t_start) < deadline - 5.0:
        improved = False
        # Sort codewords by how "safe" they are to remove
        # Safety = min coverage count among its neighbors
        safety = np.zeros(len(cur_code), dtype=np.int32)
        for i, c in enumerate(cur_code):
            safety[i] = np.min(cur_cov[neighbors[c]])
        
        # Try removing in order of highest safety (most redundant)
        order = np.argsort(-safety)
        for idx in order:
            if (time.time() - t_start) > deadline - 5.0:
                break
            i = int(idx)
            c = cur_code[i]
            if safety[i] >= 2:
                cur_code.pop(i)
                cur_cov[neighbors[c]] -= 1
                cur_size -= 1
                improved = True
                break  # Restart ordering after each removal

    if cur_size < best_size:
        best_code = cur_code[:]
        best_size = cur_size

    # Phase 3: Multi-kick local search
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if (time.time() - t_start) > deadline - 10.0:
            break
        
        # Start from current best or a new greedy
        if restart == 0:
            cur_code = best_code[:]
        else:
            cur_code = greedy_cover(restart + 100)
            cov = build_cov(cur_code)
            if not np.all(cov > 0):
                cur_code, cov = greedy_repair(cur_code, cov, deadline - 20.0)
        
        cur_cov = build_cov(cur_code)
        
        # Local search: try removing k codewords and repairing
        for k in range(1, 4):
            if (time.time() - t_start) > deadline - 5.0:
                break
            for trial in range(30):
                if (time.time() - t_start) > deadline - 5.0:
                    break
                if len(cur_code) <= k:
                    break
                
                # Remove k random codewords
                remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                remove_set = set(int(ri) for ri in remove_indices)
                
                new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                new_cov = build_cov(new_code)
                
                # Repair
                new_code, new_cov = greedy_repair(new_code, new_cov, deadline - 10.0)
                
                if np.all(new_cov > 0) and len(new_code) < len(cur_code):
                    cur_code = new_code
                    cur_cov = new_cov
                    break  # Restart with improved state
            
            # Deterministic removal pass
            if (time.time() - t_start) < deadline - 5.0:
                changed = True
                while changed:
                    changed = False
                    for i in range(len(cur_code)):
                        c = cur_code[i]
                        if np.min(cur_cov[neighbors[c]]) >= 2:
                            cur_code.pop(i)
                            cur_cov[neighbors[c]] -= 1
                            changed = True
                            break

        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Phase 4: Final aggressive removal pass
    if (time.time() - t_start) < deadline - 2.0:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        changed = True
        while changed and (time.time() - t_start) < deadline - 2.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        # Fallback
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
