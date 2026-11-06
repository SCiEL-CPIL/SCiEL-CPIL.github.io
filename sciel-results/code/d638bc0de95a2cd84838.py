import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_points = 16
    # Precompute all pairs and their indices
    pairs = []
    pair_index = {}
    idx = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            pair_index[(i, j)] = idx
            pairs.append((i, j))
            idx += 1
    
    all_triples = list(itertools.combinations(range(n_points), 3))
    triple_to_pairs = {}
    for t in all_triples:
        p1 = (t[0], t[1])
        p2 = (t[0], t[2])
        p3 = (t[1], t[2])
        triple_to_pairs[t] = [pair_index[p1], pair_index[p2], pair_index[p3]]
        
    # Bit masks for triples
    triple_masks = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}

    def compute_alpha(triples_list):
        """
        Computes the independence number exactly using backtracking.
        Returns the size of the largest independent set.
        """
        if not triples_list:
            return n_points
            
        t_masks = [triple_masks[t] for t in triples_list]
        
        # Sort points by degree (number of triples containing them) to prune better?
        # For now, simple backtracking on subsets.
        # We check subsets of size k from n_points down to 1.
        
        # Optimization: If we have a set of triples, a subset S is independent
        # iff no triple is fully contained in S.
        
        # We can use a recursive function to find the max independent set.
        # Or simply iterate k downwards and check all combinations of size k.
        # With N=16, C(16, 8) = 12870. C(16, 7) = 11440.
        # Checking each combination against ~35 triples is fast.
        
        for k in range(n_points, 0, -1):
            for combo in itertools.combinations(range(n_points), k):
                combo_mask = 0
                for p in combo:
                    combo_mask |= (1 << p)
                
                is_indep = True
                for tm in t_masks:
                    if (combo_mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return k
        return 1

    # State
    pair_count = [0] * len(pairs)
    current_triples = []
    
    def add_triple(t):
        nonlocal current_triples
        ps = triple_to_pairs[t]
        for p_idx in ps:
            pair_count[p_idx] += 1
        current_triples.append(t)

    def remove_triple(t):
        nonlocal current_triples
        ps = triple_to_pairs[t]
        for p_idx in ps:
            pair_count[p_idx] -= 1
        current_triples.remove(t)

    def is_valid(t):
        ps = triple_to_pairs[t]
        for p_idx in ps:
            if pair_count[p_idx] > 0:
                return False
        return True

    # Phase 1: Greedy Initialization
    random.shuffle(all_triples)
    for t in all_triples:
        if is_valid(t):
            add_triple(t)
            
    best_alpha = compute_alpha(current_triples)
    best_triples = current_triples[:]
    
    # Phase 2: Simulated Annealing / Local Search
    # Objective: Minimize alpha.
    # We will try to improve alpha.
    
    T = 1.0
    cooling = 0.99
    
    # To speed up, we don't recompute alpha every single step.
    # We recompute alpha when we have a "promising" move or every N steps.
    
    steps = 0
    no_improve_steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Random Move
        r = random.random()
        
        if r < 0.3:
            # Add a random triple
            t = random.choice(all_triples)
            if t not in current_triples and is_valid(t):
                add_triple(t)
                # Check alpha occasionally
                if steps % 10 == 0:
                    alpha = compute_alpha(current_triples)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = current_triples[:]
                        no_improve_steps = 0
                        if best_alpha <= 5:
                            break
                    else:
                        no_improve_steps += 1
                # Accept if it increases density or maintains it
                # Here we just accept valid adds
            else:
                # If invalid, reject
                pass
                
        elif r < 0.6:
            # Remove a random triple
            if current_triples:
                t = random.choice(current_triples)
                remove_triple(t)
                # We don't necessarily want to reduce density, but it might allow better structure
                # We will rely on the add moves to refill
                
        else:
            # Swap: Remove one, add another
            if current_triples:
                t_remove = random.choice(current_triples)
                # Try to find a valid replacement
                # To be faster, we just pick a random candidate
                t_add = random.choice(all_triples)
                
                # Temporarily remove t_remove
                ps_remove = triple_to_pairs[t_remove]
                for p_idx in ps_remove:
                    pair_count[p_idx] -= 1
                current_triples.remove(t_remove)
                
                if is_valid(t_add):
                    add_triple(t_add)
                    # Accepted
                    if steps % 10 == 0:
                        alpha = compute_alpha(current_triples)
                        if alpha < best_alpha:
                            best_alpha = alpha
                            best_triples = current_triples[:]
                            no_improve_steps = 0
                            if best_alpha <= 5:
                                break
                        else:
                            no_improve_steps += 1
                else:
                    # Revert
                    current_triples.append(t_remove)
                    for p_idx in ps_remove:
                        pair_count[p_idx] += 1
                
        # Cooling
        T *= cooling
        
        # If stuck for too long, restart or perturb
        if no_improve_steps > 500:
            # Perturbation: Randomly remove 5 triples
            for _ in range(5):
                if current_triples:
                    t = random.choice(current_triples)
                    remove_triple(t)
            no_improve_steps = 0
            T = 1.0 # Reset temperature

    return best_triples
