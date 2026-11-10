"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def fast_eval(positions, values):
        """
        Evaluate the number of non-zero terms in the square of the polynomial
        defined by positions and values.
        """
        max_deg = max(positions)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for p, v in zip(positions, values):
            dense[p] = v
        sq = np.convolve(dense, dense, mode='full')
        return int(np.count_nonzero(sq))

    def sum_collision_count(positions):
        """Count distinct sum positions p_i + p_j for i <= j"""
        sums = set()
        for i in range(12):
            for j in range(i, 12):
                sums.add(positions[i] + positions[j])
        return len(sums)

    def generate_initial_candidates():
        """
        Generate a diverse set of initial position candidates.
        Focus on structures that maximize sum collisions.
        """
        candidates = set()
        
        # 1. Arithmetic Progressions (AP)
        for start in range(0, 41):
            for step in range(1, 5):
                pos = tuple(start + i * step for i in range(12))
                if pos[-1] <= 40:
                    candidates.add(pos)
        
        # 2. Two interleaved APs
        for s1 in range(0, 30):
            for s2 in range(s1 + 1, 30):
                for step in range(1, 5):
                    pos = []
                    for i in range(6):
                        pos.append(s1 + i * step)
                        pos.append(s2 + i * step)
                    pos.sort()
                    if len(set(pos)) == 12 and pos[-1] <= 40:
                        candidates.add(tuple(pos))
        
        # 3. "Block" structures: clusters of consecutive integers
        for _ in range(2000):
            n_blocks = random.choice([2, 3, 4, 6])
            block_sizes = [12 // n_blocks] * n_blocks
            pos = []
            current_base = 0
            for size in block_sizes:
                gap = random.randint(0, 10)
                current_base += gap
                block = list(range(current_base, current_base + size))
                pos.extend(block)
                current_base += size
            if len(set(pos)) == 12 and max(pos) <= 40:
                candidates.add(tuple(sorted(pos)))
        
        # 4. Random sets
        for _ in range(2000):
            pos = tuple(sorted(random.sample(range(41), 12)))
            candidates.add(pos)
            
        return list(candidates)

    def optimize_values_for_positions(positions, time_limit=0.5):
        """
        For a fixed set of positions, find values that minimize the non-zero count.
        Uses a mix of template-based initialization and local search.
        """
        start_t = time.time()
        best_score = 1000
        best_values = None
        
        def try_values(vals):
            nonlocal best_score, best_values
            score = fast_eval(positions, vals)
            if score < best_score:
                best_score = score
                best_values = vals[:]
                return True
            return False
        
        # 1. Templates
        try_values([1] * 12)
        try_values([1 if i % 2 == 0 else -1 for i in range(12)])
        try_values([1 if i % 3 == 0 else -1 for i in range(12)])
        
        # 2. Random search with bias toward small integers
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 2000:
            n_trials += 1
            vals = []
            for _ in range(12):
                r = random.random()
                if r < 0.5:
                    mag = 1
                elif r < 0.8:
                    mag = random.randint(1, 3)
                else:
                    mag = random.randint(1, 8)
                sign = 1 if random.random() < 0.5 else -1
                vals.append(sign * mag)
            
            if try_values(vals):
                if best_score <= 12:
                    break
        
        # 3. Local search on values (if we have a good start)
        if best_values:
            current_vals = best_values[:]
            current_score = best_score
            improved = True
            while improved and time.time() - start_t < time_limit:
                improved = False
                for _ in range(50):
                    if time.time() - start_t > time_limit:
                        break
                    idx = random.randint(0, 11)
                    old_val = current_vals[idx]
                    r = random.random()
                    if r < 0.4:
                        new_val = -old_val
                    elif r < 0.7:
                        new_val = random.choice([1, -1]) * random.randint(1, 4)
                    else:
                        new_val = random.choice([1, -1]) * random.randint(1, 16)
                    
                    if new_val == old_val:
                        continue
                    
                    current_vals[idx] = new_val
                    score = fast_eval(positions, current_vals)
                    if score < current_score:
                        current_score = score
                        best_score = score
                        best_values = current_vals[:]
                        improved = True
                    else:
                        current_vals[idx] = old_val
                    if current_score <= 12:
                        return best_score, best_values
        
        return best_score, best_values

    def local_search_full(terms, time_limit=2.0):
        """Local search starting from given terms."""
        start_t = time.time()
        current_terms = list(terms)
        positions = [t[0] for t in current_terms]
        values = [t[1] for t in current_terms]
        current_score = fast_eval(positions, values)
        
        def neighbor(terms):
            terms = list(terms)
            move_type = random.random()
            if move_type < 0.4:
                idx = random.randint(0, 11)
                old_pos = terms[idx][0]
                new_pos = random.randint(0, 40)
                if new_pos == old_pos:
                    return terms
                for i, (pos, val) in enumerate(terms):
                    if i != idx and pos == new_pos:
                        return terms
                terms[idx] = (new_pos, terms[idx][1])
            elif move_type < 0.7:
                idx = random.randint(0, 11)
                new_val = random.choice([1, -1]) * random.randint(1, 64)
                terms[idx] = (terms[idx][0], new_val)
            elif move_type < 0.9:
                idx = random.randint(0, 11)
                terms[idx] = (terms[idx][0], -terms[idx][1])
            else:
                i, j = random.sample(range(12), 2)
                terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
            return terms
        
        while time.time() - start_t < time_limit:
            improved = False
            for _ in range(50):
                if time.time() - start_t > time_limit:
                    break
                new_terms = neighbor(current_terms)
                new_pos = [t[0] for t in new_terms]
                new_vals = [t[1] for t in new_terms]
                new_score = fast_eval(new_pos, new_vals)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
            if not improved:
                break
        
        return current_terms, current_score

    # Phase 1: Structural search
    candidates = generate_initial_candidates()
    candidates.sort(key=sum_collision_count)
    
    best_terms = None
    best_score = 1000
    
    n_candidates = min(500, len(candidates))
    time_per_candidate = 40.0 / n_candidates
    
    for idx, pos in enumerate(candidates[:n_candidates]):
        if time.time() > deadline - 10:
            break
            
        remaining_time = deadline - time.time() - 10
        if remaining_time <= 0:
            break
        
        rank_factor = 1.0 / (1.0 + idx * 0.1)
        t_limit = min(1.0, max(0.02, time_per_candidate * rank_factor))
        
        score, values = optimize_values_for_positions(pos, time_limit=t_limit)
        
        if values is not None and score < best_score:
            best_score = score
            best_terms = list(zip(pos, values))
            if best_score <= 12:
                break
    
    # Phase 2: Local search on the best found solution (positions + values)
    if best_terms:
        remaining = deadline - time.time()
        if remaining > 3:
            refined, refined_score = local_search_full(best_terms, time_limit=min(remaining - 2, 10.0))
            if refined_score < best_score:
                best_terms = refined
                best_score = refined_score
    
    # Phase 3: Random restarts with local search
    remaining = deadline - time.time()
    if remaining > 5:
        n_restarts = min(50, int(remaining / 2))
        for _ in range(n_restarts):
            if time.time() > deadline - 3:
                break
            positions = random.sample(range(41), 12)
            positions.sort()
            values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
            terms = list(zip(positions, values))
            
            t_limit = max(0.5, (deadline - 3 - time.time()) / max(1, n_restarts - _))
            refined, score = local_search_full(terms, time_limit=t_limit)
            if score < best_score:
                best_score = score
                best_terms = refined
                if best_score <= 12:
                    break
    
    if best_terms is None:
        # Fallback
        positions = list(range(0, 41, 4))[:12]
        values = [1] * 12
        best_terms = list(zip(positions, values))
        best_score = fast_eval(positions, values)
    
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val
    
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
    
    return coeff_list
# EVOLVE-BLOCK-END
