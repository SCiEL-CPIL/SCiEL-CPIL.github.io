import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    
    # Precompute all triples
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Map triple index to mask
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[i] = mask
            
    # Precompute conflicts
    # conflicts[i] is a set of indices j such that triple i and j share >= 2 points
    conflicts = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            # Count bits in intersection
            # Using bin(x).count('1') is fast enough for 560*560/2 ~ 150k checks
            if bin(mi & mj).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute masks by size for alpha calculation
    # We only need to check sizes down to 1, but typically alpha will be 4 or 5.
    # Checking size 16 down to 1 is too slow if we do it naively every time.
    # We will check specific sizes: 5, 4, 3, 2, 1.
    # If no independent set of size 5, check 4, etc.
    
    masks_by_size = {k: [] for k in range(1, 6)}
    for mask in range(1 << n):
        k = bin(mask).count('1')
        if 1 <= k <= 5:
            masks_by_size[k].append(mask)
            
    def calc_alpha_and_witness(edges_set):
        """
        Returns (alpha, witness_mask).
        If no independent set of size k is found, returns (k-1, None) and continues.
        If we find one of size k, we return it immediately? 
        Actually, we want the MAXIMUM size. So we check from high to low.
        If we find an independent set of size 5, alpha is 5.
        If not, check 4.
        """
        if not edges_set:
            return 16, (1 << 16) - 1
            
        # Check sizes 5 down to 1
        for k in [5, 4, 3, 2, 1]:
            for m in masks_by_size[k]:
                is_indep = True
                for e_idx in edges_set:
                    e_mask = triple_masks[e_idx]
                    if (m & e_mask) == e_mask:
                        is_indep = False
                        break
                if is_indep:
                    return k, m
        return 1, 0

    def is_valid_add(triple_idx, edges_set):
        if triple_idx in edges_set:
            return False
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True

    def greedy_pack():
        current_triples = set()
        available = list(range(n_triples))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        
        return current_triples

    def get_triples_in_mask(mask):
        """
        Returns list of triple indices fully contained in the given mask.
        """
        res = []
        for i in range(n_triples):
            if (triple_masks[i] & mask) == triple_masks[i]:
                res.append(i)
        return res

    best_alpha = 17
    best_solution = None
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random.seed(1234 + restart)
        
        current_triples = greedy_pack()
        current_alpha, witness = calc_alpha_and_witness(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_solution = set(current_triples)
            if best_alpha <= 3: # Theoretical min for n=16 is likely 4 or 5, 3 is very hard
                break
        
        # Local Search
        steps = 0
        max_steps = 2000
        while steps < max_steps:
            if time.time() > deadline - 5:
                break
            steps += 1
            
            # 1. Targeted Move: Try to break the current witness
            if witness is not None and current_alpha > 1:
                # Get all triples inside the witness
                triples_in_witness = get_triples_in_mask(witness)
                random.shuffle(triples_in_witness)
                
                move_found = False
                for t_idx in triples_in_witness:
                    if t_idx in current_triples:
                        continue
                    
                    # Try direct add
                    if is_valid_add(t_idx, current_triples):
                        new_set = set(current_triples)
                        new_set.add(t_idx)
                        new_alpha, new_witness = calc_alpha_and_witness(new_set)
                        if new_alpha <= current_alpha:
                            current_triples = new_set
                            current_alpha = new_alpha
                            witness = new_witness
                            move_found = True
                            break
                    else:
                        # Try swap: remove one conflict, add t_idx
                        # Find conflicts of t_idx that are in current_triples
                        conflicts_in_set = [c for c in conflicts[t_idx] if c in current_triples]
                        random.shuffle(conflicts_in_set)
                        
                        for c_idx in conflicts_in_set:
                            new_set = set(current_triples)
                            new_set.remove(c_idx)
                            # Check if t_idx is valid against new_set
                            if is_valid_add(t_idx, new_set):
                                new_set.add(t_idx)
                                new_alpha, new_witness = calc_alpha_and_witness(new_set)
                                if new_alpha <= current_alpha:
                                    current_triples = new_set
                                    current_alpha = new_alpha
                                    witness = new_witness
                                    move_found = True
                                    break
                        if move_found:
                            break
                
            # 2. Random Move: If no targeted move or to explore
            if not move_found:
                if not current_triples:
                    # Add random
                    idx = random.randint(0, n_triples - 1)
                    if is_valid_add(idx, current_triples):
                        current_triples.add(idx)
                    continue
                
                # Pick a triple to remove
                rem_idx = random.choice(list(current_triples))
                current_triples.remove(rem_idx)
                
                # Try to add a better triple
                # Sample candidates
                found_add = False
                candidates = [random.randint(0, n_triples - 1) for _ in range(20)]
                candidates = list(set(candidates))
                
                for cand in candidates:
                    if cand in current_triples:
                        continue
                    if cand == rem_idx:
                        continue
                    if is_valid_add(cand, current_triples):
                        current_triples.add(cand)
                        found_add = True
                        break
                
                if not found_add:
                    # Restore
                    current_triples.add(rem_idx)
                else:
                    # Evaluate new state
                    new_alpha, new_witness = calc_alpha_and_witness(current_triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        witness = new_witness
                    elif new_alpha == current_alpha:
                        witness = new_witness
                        # Accept with some probability to escape local minima? 
                        # For now, accept if it's not worse.
                    else:
                        # Revert
                        current_triples.remove(cand)
                        current_triples.add(rem_idx)
                        # Recalculate alpha? It's the same as before
                        # current_alpha, witness = calc_alpha_and_witness(current_triples) # Optimization: skip if reverted to exact same set
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_solution = set(current_triples)
                if best_alpha <= 3:
                    break
                    
    if best_solution is None:
        return [[0, 1, 2]]
        
    return [list(all_triples[i]) for i in best_solution]
