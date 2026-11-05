import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    N = len(all_triples)
    
    # Precompute triple masks and pairs
    triple_mask = [0] * N
    triple_pairs = [[] for _ in range(N)]
    for i, t in enumerate(all_triples):
        triple_mask[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_pairs[i] = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]

    # Build conflict graph constraints for MILP
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    rows = []
    cols = []
    data = []
    
    for p, t_list in pair_to_triples.items():
        m = len(t_list)
        for k in range(m):
            for l in range(k+1, m):
                i = t_list[k]
                j = t_list[l]
                rows.append(len(rows))
                cols.append(i)
                data.append(1)
                rows.append(len(rows))
                cols.append(j)
                data.append(1)
                
    A = csr_matrix((data, (rows, cols)), shape=(len(rows), N))
    c = -np.ones(N)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)
    cons = LinearConstraint(A, ub=1)
    
    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        if res.success:
            x = res.x
            active_indices = [i for i in range(N) if x[i] > 0.5]
        else:
            active_indices = []
    except Exception:
        active_indices = []
        
    if not active_indices:
        random.seed(42)
        current = []
        used_pairs = set()
        indices = list(range(N))
        random.shuffle(indices)
        for idx in indices:
            compatible = True
            for p in triple_pairs[idx]:
                if p in used_pairs:
                    compatible = False
                    break
            if compatible:
                current.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
        active_indices = current

    def compute_alpha_fast(active_idx):
        if not active_idx:
            return 16
        masks = [triple_mask[i] for i in active_idx]
        best = 0
        # Backtracking to find max independent set
        # Order points to help pruning? Random or fixed. Fixed is fine.
        def backtrack(idx, current_mask, current_size):
            nonlocal best
            if current_size > best:
                best = current_size
            if idx == 16:
                return
            # Exclude idx
            backtrack(idx + 1, current_mask, current_size)
            # Include idx
            new_mask = current_mask | (1 << idx)
            valid = True
            for m in masks:
                if (m & new_mask) == m:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, new_mask, current_size + 1)
        backtrack(0, 0, 0)
        return best

    current_alpha = compute_alpha_fast(active_indices)
    best_alpha = current_alpha
    best_indices = list(active_indices)
    
    # Local Search
    def is_compatible(idx, used_pairs):
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True
        
    used_pairs = set()
    for idx in active_indices:
        for p in triple_pairs[idx]:
            used_pairs.add(p)
            
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > 50:
            break
            
        # Find max independent set (we need the set to target)
        # Reuse backtracking to also store the set? 
        # For now, just compute alpha. If we want to add a triple, we need to know which points are in the max ind set.
        # Let's modify compute_alpha_fast to return the set as well? 
        # Or just pick random triples from the whole space that are compatible?
        # Better: Pick random compatible triple from the whole space.
        
        # Generate random candidate
        candidates = [i for i in range(N) if is_compatible(i, used_pairs)]
        
        if not candidates:
            # Try swap
            if active_indices:
                old_idx = random.choice(active_indices)
                old_pairs = triple_pairs[old_idx]
                active_indices.remove(old_idx)
                for p in old_pairs:
                    used_pairs.discard(p)
                
                new_candidates = [i for i in range(N) if is_compatible(i, used_pairs)]
                if new_candidates:
                    new_idx = random.choice(new_candidates)
                    active_indices.append(new_idx)
                    for p in triple_pairs[new_idx]:
                        used_pairs.add(p)
                else:
                    active_indices.append(old_idx)
                    for p in old_pairs:
                        used_pairs.add(p)
            else:
                break
                
        else:
            new_idx = random.choice(candidates)
            active_indices.append(new_idx)
            for p in triple_pairs[new_idx]:
                used_pairs.add(p)
                
        new_alpha = compute_alpha_fast(active_indices)
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_indices = list(active_indices)
            no_improve = 0
        else:
            no_improve += 1
            # Revert? 
            # If we added, revert to keep search local?
            # Let's revert if no improvement to avoid drift.
            if new_idx in active_indices:
                active_indices.remove(new_idx)
                for p in triple_pairs[new_idx]:
                    used_pairs.discard(p)
            # Note: If we did a swap, we already updated active_indices. 
            # We should track if we did a swap. 
            # Simplification: Only do additions and revert if no improvement.
            # Swaps are complex to revert in this simple loop.
            # Let's stick to additions only for now, or just accept the state.
            # Actually, adding triples *always* decreases or maintains alpha.
            # So if we add a compatible triple, alpha cannot increase.
            # So new_alpha <= current_alpha.
            # If new_alpha == current_alpha, we didn't improve, but we added a triple.
            # This might help future additions?
            # Let's keep the addition.
            
    result = [list(all_triples[i]) for i in best_indices]
    return result
