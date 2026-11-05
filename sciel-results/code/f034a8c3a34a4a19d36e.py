import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors for radius=1
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

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 30.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, max_time=None):
        if max_time is None:
            max_time = deadline - 10.0
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

    def compress(code, cov, max_time=None):
        """Greedy deletion: remove codewords whose removal doesn't break coverage."""
        if max_time is None:
            max_time = deadline - 5.0
        cur_code = code[:]
        cur_cov = cov.copy()
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            for i in range(len(cur_code)):
                if (time.time() - t_start) > max_time:
                    break
                c = cur_code[i]
                # Check if any neighbor is only covered by this codeword
                min_cov = np.min(cur_cov[neighbors[c]])
                if min_cov >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    improved = True
                    break
        return cur_code, cur_cov

    def swap_local_search(code, cov, max_time=None, rng=None):
        """1-swap local search: try replacing one codeword with another to reduce size or maintain size while improving coverage slack."""
        if max_time is None:
            max_time = deadline - 10.0
        if rng is None:
            rng = np.random.default_rng(123)
        cur_code = code[:]
        cur_cov = cov.copy()
        best_size = len(cur_code)
        
        # Try to reduce size via swaps: remove one, add another that covers more uncovered
        # First, try pure deletions
        cur_code, cur_cov = compress(cur_code, cur_cov, max_time)
        best_size = len(cur_code)
        
        # Now try 1-swap: for each codeword, try replacing with a non-codeword
        # that covers at least as much as the one being removed
        for iteration in range(200):
            if (time.time() - t_start) > max_time:
                break
            if len(cur_code) == 0:
                break
            
            # Pick a random codeword to try removing
            i = rng.integers(0, len(cur_code))
            c = cur_code[i]
            
            # Temporarily remove
            cur_code.pop(i)
            cur_cov[neighbors[c]] -= 1
            
            # Find uncovered words
            uncovered = cur_cov == 0
            if not np.any(uncovered):
                # We can remove this codeword for free
                best_size = len(cur_code)
                continue
            
            # Find best replacement
            gains = np.sum(uncovered[neighbors], axis=1)
            # Exclude current codewords
            for cc in cur_code:
                gains[cc] = -1
            
            best = np.argmax(gains)
            
            if gains[best] >= np.sum(uncovered):
                # This replacement covers all uncovered
                cur_code.append(int(best))
                cur_cov[neighbors[best]] += 1
            else:
                # Revert
                cur_code.insert(i, c)
                cur_cov[neighbors[c]] += 1
        
        # Final compression
        cur_code, cur_cov = compress(cur_code, cur_cov, max_time)
        return cur_code, cur_cov

    def simulated_annealing(code, cov, max_time=None):
        """Simulated annealing with 1-swap and 1-delete moves."""
        if max_time is None:
            max_time = deadline - 10.0
        rng = np.random.default_rng(456)
        cur_code = code[:]
        cur_cov = cov.copy()
        cur_energy = len(cur_code)
        best_code = cur_code[:]
        best_energy = cur_energy
        
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        while T > T_min and (time.time() - t_start) < max_time:
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.4 and len(cur_code) > 0:
                # Try deleting a codeword
                i = rng.integers(0, len(cur_code))
                c = cur_code[i]
                cur_code.pop(i)
                cur_cov[neighbors[c]] -= 1
                uncovered_count = np.sum(cur_cov == 0)
                new_energy = len(cur_code) + 10.0 * uncovered_count
                delta = new_energy - cur_energy
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    cur_energy = new_energy
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                        best_code = cur_code[:]
                else:
                    # Revert
                    cur_code.insert(i, c)
                    cur_cov[neighbors[c]] += 1
                    
            elif move_type < 0.8 and len(cur_code) > 0:
                # Try swapping: remove one, add another
                i = rng.integers(0, len(cur_code))
                c = cur_code[i]
                cur_code.pop(i)
                cur_cov[neighbors[c]] -= 1
                
                # Pick a random candidate to add
                candidates = np.setdiff1d(np.arange(space_size), cur_code)
                if len(candidates) > 0:
                    new_c = candidates[rng.integers(0, len(candidates))]
                    cur_code.append(int(new_c))
                    cur_cov[neighbors[new_c]] += 1
                    uncovered_count = np.sum(cur_cov == 0)
                    new_energy = len(cur_code) + 10.0 * uncovered_count
                    delta = new_energy - cur_energy
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        cur_energy = new_energy
                        if cur_energy < best_energy:
                            best_energy = cur_energy
                            best_code = cur_code[:]
                    else:
                        # Revert
                        cur_code.pop()
                        cur_cov[neighbors[new_c]] -= 1
                        cur_code.insert(i, c)
                        cur_cov[neighbors[c]] += 1
            else:
                # Try adding a codeword (to fix coverage issues)
                uncovered = cur_cov == 0
                if np.any(uncovered):
                    gains = np.sum(uncovered[neighbors], axis=1)
                    # Weighted random selection
                    probs = gains / (gains.sum() + 1e-10)
                    new_c = rng.choice(space_size, p=probs)
                    cur_code.append(int(new_c))
                    cur_cov[neighbors[new_c]] += 1
                    uncovered_count = np.sum(cur_cov == 0)
                    new_energy = len(cur_code) + 10.0 * uncovered_count
                    delta = new_energy - cur_energy
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        cur_energy = new_energy
                        if cur_energy < best_energy:
                            best_energy = cur_energy
                            best_code = cur_code[:]
                    else:
                        cur_code.pop()
                        cur_cov[neighbors[new_c]] -= 1
            
            T *= cooling
        
        # Ensure best_code is valid
        best_cov = build_cov(best_code)
        if not np.all(best_cov > 0):
            best_code, best_cov = greedy_repair(best_code, best_cov, max_time)
        
        return best_code, best_cov

    def iterated_local_search(code, cov, max_time=None):
        """ILS: kick + local search (swap + compress) + acceptance."""
        if max_time is None:
            max_time = deadline - 5.0
        rng = np.random.default_rng(789)
        cur_code = code[:]
        cur_cov = cov.copy()
        best_code = cur_code[:]
        best_size = len(cur_code)
        
        for iteration in range(1000):
            if (time.time() - t_start) > max_time:
                break
            
            # Kick: random perturbation
            kick_size = max(1, len(cur_code) // 10)
            kick_size = min(kick_size, 5)
            
            # Remove some random codewords
            indices = rng.choice(len(cur_code), size=min(kick_size, len(cur_code)), replace=False)
            for i in sorted(indices, reverse=True):
                c = cur_code.pop(i)
                cur_cov[neighbors[c]] -= 1
            
            # Repair coverage
            cur_code, cur_cov = greedy_repair(cur_code, cur_cov, max_time)
            
            # Local search
            cur_code, cur_cov = swap_local_search(cur_code, cur_cov, max_time, rng)
            
            # Acceptance
            if len(cur_code) < best_size:
                best_size = len(cur_code)
                best_code = cur_code[:]
        
        return best_code, best_cov

    # === MAIN ALGORITHM ===
    
    # Phase 1: Multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(6):
        if (time.time() - t_start) > deadline - 50.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 45.0)
        code, cov = compress(code, cov, deadline - 40.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Iterated local search with swap moves
    if (time.time() - t_start) < deadline - 30.0 and best_code is not None:
        cov = build_cov(best_code)
        best_code, cov = iterated_local_search(best_code, cov, deadline - 25.0)
        if len(best_code) < best_size:
            best_size = len(best_code)

    # Phase 3: Simulated annealing refinement
    if (time.time() - t_start) < deadline - 15.0 and best_code is not None:
        cov = build_cov(best_code)
        best_code, cov = simulated_annealing(best_code, cov, deadline - 10.0)
        if len(best_code) < best_size:
            best_size = len(best_code)

    # Phase 4: Final compression
    if (time.time() - t_start) < deadline - 5.0 and best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        cur_code, cur_cov = compress(cur_code, cur_cov, deadline - 2.0)
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        best_code, final_cov = greedy_repair(best_code, final_cov, deadline - 1.0)
        if not np.all(final_cov > 0):
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
