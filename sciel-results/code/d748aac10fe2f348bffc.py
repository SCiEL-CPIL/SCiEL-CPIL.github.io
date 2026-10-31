import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_score = 1.0
    best_set = [0, 1, 3]
    
    def compute_score(U):
        if not U or 0 not in U or len(U) < 2:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Vectorized scoring using numpy
        arr = np.array(U)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    def simulated_annealing(n_particles, max_val, T_init=10.0, T_final=0.1, n_iter=5000):
        """
        Lattice gas model: each element is a particle on positions 0..max_val.
        Particle 0 is fixed at position 0.
        Energy = -log(|U-U|/|U+U|) + beta * log(2*max(U)+1)
        We minimize this energy via SA.
        """
        nonlocal best_score, best_set
        
        # Initialize: particle 0 at 0, others random
        positions = np.zeros(n_particles, dtype=int)
        positions[0] = 0
        positions[1:] = np.random.randint(1, max_val + 1, n_particles - 1)
        
        # Ensure uniqueness
        for i in range(1, n_particles):
            if positions[i] in positions[:i]:
                positions[i] = i  # fallback
        
        positions = np.sort(positions)
        
        def energy(pos):
            """Compute the negative of our score (to minimize)."""
            if len(pos) < 2 or pos[0] != 0:
                return 1e6
            max_u = pos[-1]
            if max_u == 0:
                return 1e6
            
            sums = pos[:, None] + pos[None, :]
            diffs = pos[:, None] - pos[None, :]
            
            n_sum = len(np.unique(sums))
            n_diff = len(np.unique(diffs))
            
            if n_sum == 0:
                return 1e6
            
            base = 2 * max_u + 1
            if base <= 1:
                return 1e6
            
            ratio = n_diff / n_sum
            if ratio <= 1:
                return 1.0
            
            score_val = 1 + math.log(ratio) / math.log(base)
            return -score_val  # Minimize negative score
        
        curr_energy = energy(positions)
        best_pos = positions.copy()
        best_e = curr_energy
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule (exponential cooling)
            T = T_init * (T_final / T_init) ** (it / n_iter)
            
            # Propose a move: pick a particle (not 0) and change its position
            if n_particles > 1:
                idx = random.randint(1, n_particles - 1)
                old_pos = positions[idx]
                
                # Propose new position
                delta = random.randint(-5, 5)
                new_pos = old_pos + delta
                
                if new_pos <= 0 or new_pos > max_val:
                    continue
                
                # Check if position is already taken
                if new_pos in positions:
                    continue
                
                positions[idx] = new_pos
                new_energy = energy(positions)
                
                # Accept or reject
                if new_energy < curr_energy:
                    curr_energy = new_energy
                    if new_energy < best_e:
                        best_e = new_energy
                        best_pos = positions.copy()
                        # Check if this is our global best
                        cand = sorted(best_pos.tolist())
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                else:
                    # Metropolis acceptance
                    delta_e = new_energy - curr_energy
                    if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                        curr_energy = new_energy
                    else:
                        positions[idx] = old_pos
            else:
                break
        
        return best_pos.tolist()

    # Phase 1: Try various sizes with SA
    for n in range(3, 12):
        for max_val in [20, 30, 50, 80, 120]:
            if time.time() > deadline - 10:
                break
            result = simulated_annealing(n, max_val, T_init=5.0, T_final=0.05, n_iter=3000)
            if result and len(result) >= 2:
                try_set(result)

    # Phase 2: Local search refinement on best found
    def local_search(start_set, max_iters=3000):
        nonlocal best_score, best_set
        curr = sorted(set(start_set))
        if not curr or 0 not in curr or len(curr) < 2:
            return curr
        
        curr_score = compute_score(curr)
        no_improve_count = 0
        max_no_improve = 15
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            n_curr = len(curr)
            max_cur = max(curr)
            
            # Try adding an element
            if n_curr < 15:
                search_limit = max_cur + 15
                for new_el in range(1, search_limit + 1):
                    if new_el in curr:
                        continue
                    cand = sorted(curr + [new_el])
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        break
            
            # Try replacing an element
            if n_curr >= 2 and not improved:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    search_limit = max_cur + 15
                    for new_el in range(1, search_limit + 1):
                        if new_el in curr:
                            continue
                        cand = list(curr)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            improved = True
                            no_improve_count = 0
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            break
                    if improved:
                        break
            
            # Try removing an element
            if n_curr > 2 and not improved:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(curr) if j != i]
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        break
            
            if improved:
                continue
            
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
            
            # Random perturbation
            if random.random() < 0.1:
                n = random.randint(3, min(12, n_curr + 2))
                max_m = max(15, max_cur + random.randint(0, 10))
                if n - 1 > 0:
                    inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
                    cand = sorted([0] + inner)
                else:
                    cand = [0]
                if len(cand) >= 2:
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
        
        return curr

    # Refine the best set found
    if best_score > 1.0:
        local_search(best_set)

    # Random restarts with SA
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        n = random.randint(4, 10)
        max_m = random.randint(30, 100)
        result = simulated_annealing(n, max_m, T_init=8.0, T_final=0.1, n_iter=2000)
        if result and len(result) >= 2:
            try_set(result)
            # Quick local refinement
            if compute_score(result) > best_score - 0.01:
                local_search(result, max_iters=1000)

    return list(best_set)
