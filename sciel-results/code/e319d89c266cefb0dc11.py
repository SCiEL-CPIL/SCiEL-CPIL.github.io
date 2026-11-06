import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples = []
    triple_to_idx = {}
    for c in itertools.combinations(all_points, 3):
        mask = 0
        for p in c:
            mask |= (1 << p)
        idx = len(all_triples)
        all_triples.append((c, mask))
        triple_to_idx[c] = idx
        
    num_triples = len(all_triples)
    
    # Precompute conflict matrix
    # conflict[i][j] is True if triple i and j share > 1 point
    conflict = [[False]*num_triples for _ in range(num_triples)]
    for i in range(num_triples):
        mi = all_triples[i][1]
        for j in range(i+1, num_triples):
            mj = all_triples[j][1]
            inter = mi & mj
            if inter & (inter - 1): # popcount > 1
                conflict[i][j] = True
                conflict[j][i] = True

    def can_add(active_set, idx):
        for j in active_set:
            if conflict[idx][j]:
                return False
        return True
        
    def get_conflicts(active_set, idx):
        return [j for j in active_set if conflict[idx][j]]

    def find_independent_set_size_6(active_indices):
        active_masks = [all_triples[i][1] for i in active_indices]
        for s in itertools.combinations(all_points, 6):
            smask = 0
            for p in s:
                smask |= (1 << p)
            is_independent = True
            for tm in active_masks:
                if (tm & smask) == tm:
                    is_independent = False
                    break
            if is_independent:
                return s
        return None

    def generate_random_packing():
        active = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if can_add(active, idx):
                active.add(idx)
        return active

    best_active = None
    best_alpha = 16
    
    while time.time() < deadline:
        if best_active is None:
            active = generate_random_packing()
        else:
            # Check if current best is already optimal (alpha <= 5)
            ind_set = find_independent_set_size_6(best_active)
            if ind_set is None:
                break # Found optimal alpha=5
            
            # Try to break the independent set
            # Pick a random triple from ind_set
            sub_triples = list(itertools.combinations(ind_set, 3))
            random.shuffle(sub_triples)
            target_triple = sub_triples[0]
            target_idx = triple_to_idx[target_triple]
            
            # Create a copy to try the move
            trial_active = set(best_active)
            
            # Remove conflicts
            conflicts = get_conflicts(trial_active, target_idx)
            for c_idx in conflicts:
                trial_active.remove(c_idx)
            
            # Add target
            trial_active.add(target_idx)
            
            # Refill: try to add random valid triples to increase density
            # We want to maximize the number of triples to minimize alpha
            indices = list(range(num_triples))
            random.shuffle(indices)
            added_count = 0
            for idx in indices:
                if idx not in trial_active and can_add(trial_active, idx):
                    trial_active.add(idx)
                    added_count += 1
                    # Don't refill too aggressively if we are just testing a move?
                    # Actually, we want the maximal packing resulting from this move.
                    # So refill fully.
            
            # Check if this move improved the situation
            # We don't need exact alpha, just check if ind_set of size 6 exists
            new_ind_set = find_independent_set_size_6(trial_active)
            
            # Heuristic: Accept if we broke the specific independent set or if the new system looks "better"
            # A simple metric: number of triples (more triples -> likely lower alpha)
            # Or just accept if new_ind_set is None (immediate win)
            # Or accept if len(trial_active) >= len(best_active) (more coverage)
            
            if new_ind_set is None:
                best_active = trial_active
                break
            elif len(trial_active) > len(best_active):
                # More triples, likely better
                best_active = trial_active
            elif random.random() < 0.1:
                # Occasional random acceptance to escape local minima
                best_active = trial_active

    if best_active is None:
        best_active = generate_random_packing()
        
    triples = [list(all_triples[i][0]) for i in best_active]
    return triples

if __name__ == "__main__":
    res = construct()
    print(f"Constructed {len(res)} triples")
    # Verify linearity
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                raise ValueError(f"Triples {i} and {j} share more than 1 point: {inter}")
    print("Linearity valid")
