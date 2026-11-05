import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    n_points = 16
    
    # Precompute all triples
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute pairs for each triple
    triple_pairs = [ [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples ]
    
    # Precompute triple bitmask (16-bit)
    triple_mask = [0] * n_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_mask[i] = mask
    
    # Precompute all 8-subsets as bitmasks
    all_8subsets = list(combinations(points, 8))
    n_8subsets = len(all_8subsets)
    s8_mask = [0] * n_8subsets
    for j, s8 in enumerate(all_8subsets):
        mask = 0
        for p in s8:
            mask |= (1 << p)
        s8_mask[j] = mask
    
    # Precompute: for each triple, which 8-subsets contain it
    triple_to_8subsets = [[] for _ in range(n_triples)]
    for j in range(n_8subsets):
        m = s8_mask[j]
        for i in range(n_triples):
            if (triple_mask[i] & m) == triple_mask[i]:
                triple_to_8subsets[i].append(j)
    
    # State
    used_pairs = set()
    current_triples = []
    covered_8subsets = [0] * n_8subsets
    current_alpha = 16
    
    def get_alpha_fast():
        """Check if all 8-subsets are covered. If not, alpha >= 8."""
        for j in range(n_8subsets):
            if covered_8subsets[j] == 0:
                return 8
        return 7
    
    def get_exact_alpha():
        """Compute the exact independence number by checking all subsets from largest to smallest."""
        # Check from largest to smallest
        for k in range(16, -1, -1):
            for s in combinations(points, k):
                s_set = set(s)
                independent = True
                for t in current_triples:
                    if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    def is_valid_add(idx):
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True
    
    def add_triple(idx):
        t = all_triples[idx]
        current_triples.append(t)
        for p in triple_pairs[idx]:
            used_pairs.add(p)
        for j in triple_to_8subsets[idx]:
            covered_8subsets[j] += 1
    
    def remove_triple(idx):
        t = all_triples[idx]
        for i, ct in enumerate(current_triples):
            if ct == t:
                current_triples.pop(i)
                break
        for p in triple_pairs[idx]:
            used_pairs.discard(p)
        for j in triple_to_8subsets[idx]:
            covered_8subsets[j] -= 1
    
    def init_greedy_direct():
        """Greedy construction that directly minimizes independence number."""
        nonlocal used_pairs, current_triples, covered_8subsets, current_alpha
        used_pairs.clear()
        current_triples.clear()
        covered_8subsets = [0] * n_8subsets
        
        # Greedily add triples that maximize reduction in uncovered 8-subsets
        while True:
            best_idx = -1
            best_gain = 0
            for idx in range(n_triples):
                if not is_valid_add(idx):
                    continue
                # Count how many currently-uncovered 8-subsets this triple covers
                gain = 0
                for j in triple_to_8subsets[idx]:
                    if covered_8subsets[j] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx == -1 or best_gain == 0:
                break
            add_triple(best_idx)
        
        current_alpha = get_alpha_fast()
    
    # Initial greedy
    init_greedy_direct()
    best_alpha = current_alpha
    best_triples = list(current_triples)
    
    # Local search with simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    steps_since_improvement = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a triple
            candidates = []
            for idx in range(n_triples):
                if is_valid_add(idx):
                    candidates.append(idx)
            if not candidates:
                continue
                
            idx = random.choice(candidates)
            old_alpha = current_alpha
            
            add_triple(idx)
            new_alpha = get_alpha_fast()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, (old_alpha - new_alpha) / T + 0.1)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        elif move_type < 0.7:
            # Remove a triple
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx = triple_to_idx[t_rem]
            
            old_alpha = current_alpha
            remove_triple(idx)
            new_alpha = get_alpha_fast()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, (old_alpha - new_alpha) / T + 0.1)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                add_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        else:
            # Swap: remove one, add another
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx_rem = triple_to_idx[t_rem]
            
            remove_triple(idx_rem)
            
            candidates = []
            for idx_add in range(n_triples):
                if idx_add != idx_rem and is_valid_add(idx_add):
                    candidates.append(idx_add)
                    
            if not candidates:
                add_triple(idx_rem)
                continue
                
            idx_add = random.choice(candidates)
            
            old_alpha = current_alpha
            add_triple(idx_add)
            new_alpha = get_alpha_fast()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, (old_alpha - new_alpha) / T + 0.1)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx_add)
                add_triple(idx_rem)
                current_alpha = old_alpha
                steps_since_improvement += 1

        T *= cooling
        if T < T_min:
            T = T_min
            
        if steps_since_improvement > max_stuck:
            init_greedy_direct()
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = list(current_triples)
            steps_since_improvement = 0
            T = 1.0
    
    # Final: compute exact alpha for best_triples and verify
    used_pairs.clear()
    current_triples.clear()
    for t in best_triples:
        idx = triple_to_idx[t]
        add_triple(idx)
    
    exact_alpha = get_exact_alpha()
    
    return best_triples
