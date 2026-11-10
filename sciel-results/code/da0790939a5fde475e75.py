import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
    
    # Precompute triple masks
    triple_masks = []
    for t in all_triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    # Key data structure for fast alpha computation:
    # For each pair (u,v), store the set of vertices w such that {u,v,w} is a triple.
    # This lets us check in O(1) per pair whether adding v to a set S completes a triple.
    # pair_to_third: dict mapping (u,v) -> bitmask of w's
    pair_to_third = {}
    for i, t in enumerate(all_triples):
        u, v, w = t
        pair_to_third[(u, v)] = pair_to_third.get((u, v), 0) | (1 << w)
        pair_to_third[(v, u)] = pair_to_third.get((v, u), 0) | (1 << w)
        pair_to_third[(u, w)] = pair_to_third.get((u, w), 0) | (1 << v)
        pair_to_third[(w, u)] = pair_to_third.get((w, u), 0) | (1 << v)
        pair_to_third[(v, w)] = pair_to_third.get((v, w), 0) | (1 << u)
        pair_to_third[(w, v)] = pair_to_third.get((w, v), 0) | (1 << u)
    
    def calc_alpha(triple_indices):
        """Fast independence number using pair-based pruning."""
        if not triple_indices:
            return 16
        
        # Build the set of active triple masks
        active_masks = [triple_masks[i] for i in triple_indices]
        
        # For fast checking: when we add vertex v to current set S (as bitmask),
        # we need to check if there exist u, w in S such that {u, v, w} is an active triple.
        # Equivalent: for each u in S, check if pair_to_third[(u,v)] & S has any bit set.
        # But we also need to ensure the triple is in our active set.
        
        # Alternative: precompute for each vertex v, a list of (u, w, triple_idx) where triple is active.
        # Simpler: just use the triple_masks list directly with a smarter check.
        
        # Most efficient: for a given current_mask and new vertex v,
        # check if any active triple contains v and has the other two bits in current_mask.
        
        # Precompute: for each vertex v, list of active triple indices containing v
        v2active = [[] for _ in range(16)]
        for idx, tm in enumerate(active_masks):
            for v in range(16):
                if tm & (1 << v):
                    v2active[v].append(idx)
        
        best = [0]
        
        # Vertex ordering by degree (number of active triples containing them)
        deg = [len(v2active[v]) for v in range(16)]
        order = sorted(range(16), key=lambda v: -deg[v])
        
        def is_valid_add(current_mask, v):
            """Check if adding v to current_mask doesn't complete any active triple."""
            for idx in v2active[v]:
                tm = active_masks[idx]
                # tm contains v. Check if the other two bits are in current_mask
                others = tm & ~(1 << v)
                if (others & current_mask) == others:
                    return False
            return True
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == 16:
                return
            
            v = order[idx]
            
            # Option 1: don't include v
            search(idx + 1, current_mask, current_size)
            
            # Option 2: include v
            if is_valid_add(current_mask, v):
                new_mask = current_mask | (1 << v)
                search(idx + 1, new_mask, current_size + 1)
        
        search(0, 0, 0)
        return best[0]
    
    # Greedy construction optimized for low alpha
    random.seed(42)
    
    best_alpha = 17
    best_solution = None
    
    # Strategy: start with a maximal linear triple system, then do local search
    # to minimize alpha. The key is to have a fast alpha evaluator.
    
    def greedy_init(seed_val):
        """Generate a maximal linear triple system via greedy with randomization."""
        rng = random.Random(seed_val)
        used_pairs = set()
        current = []
        
        # Shuffle triple indices
        indices = list(range(n_triples))
        rng.shuffle(indices)
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                for p in pairs:
                    used_pairs.add(p)
                current.append(t_idx)
        
        return current
    
    def local_search(current, seed_val):
        """Simulated annealing / hill climbing to minimize alpha."""
        rng = random.Random(seed_val)
        cur_alpha = calc_alpha(current)
        best_local_alpha = cur_alpha
        best_local = list(current)
        
        used_pairs = set()
        for t_idx in current:
            for p in triple_pairs[t_idx]:
                used_pairs.add(p)
        
        # Temperature schedule
        temp = 2.0
        cooling = 0.995
        
        steps = 0
        while time.time() < deadline - 8:
            steps += 1
            
            # Pick a random triple to remove
            if not current:
                break
            remove_idx = rng.randint(0, len(current) - 1)
            t_remove = current[remove_idx]
            
            # Remove it
            for p in triple_pairs[t_remove]:
                used_pairs.discard(p)
            current.pop(remove_idx)
            
            # Try to add a random triple
            t_add = rng.randint(0, n_triples - 1)
            pairs_add = triple_pairs[t_add]
            valid = True
            for p in pairs_add:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                for p in pairs_add:
                    used_pairs.add(p)
                current.append(t_add)
                
                new_alpha = calc_alpha(current)
                
                if new_alpha <= cur_alpha:
                    cur_alpha = new_alpha
                    if new_alpha < best_local_alpha:
                        best_local_alpha = new_alpha
                        best_local = list(current)
                else:
                    # Accept with probability exp(-delta/temp)
                    import math
                    prob = math.exp(-(new_alpha - cur_alpha) / max(temp, 0.01))
                    if rng.random() < prob:
                        cur_alpha = new_alpha
                    else:
                        # Reject: revert
                        current.pop()
                        for p in pairs_add:
                            used_pairs.discard(p)
                        current.insert(remove_idx, t_remove)
                        for p in triple_pairs[t_remove]:
                            used_pairs.add(p)
            else:
                # Revert
                current.insert(remove_idx, t_remove)
                for p in triple_pairs[t_remove]:
                    used_pairs.add(p)
            
            temp *= cooling
        
        return best_local, best_local_alpha
    
    # Multiple restarts with different seeds
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        seed_val = 42 + r * 137
        current = greedy_init(seed_val)
        
        alpha = calc_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current)
        
        if best_alpha <= 4:
            break
        
        # Local search
        best_local, local_alpha = local_search(current, seed_val)
        if local_alpha < best_alpha:
            best_alpha = local_alpha
            best_solution = best_local
        
        if best_alpha <= 4:
            break
    
    if best_solution is None:
        best_solution = list(range(min(10, n_triples)))
    
    return [list(all_triples[i]) for i in best_solution]
