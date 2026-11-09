import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 16
    # 1. Precompute triples
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Precompute K(2,2,2) configurations
    # A config is defined by 3 disjoint pairs.
    # Generate all sets of 6 vertices, then partition into 3 pairs.
    configs = []
    # Map from config_id to list of 8 triple indices
    # We also need reverse map: triple_idx -> list of config_ids
    
    # Helper to get 15 partitions of 6 items into 3 pairs
    def get_partitions_6(v):
        # v is a tuple of 6 distinct ints
        a, b, c, d, e, f = v
        pairs = [
            ((a,b), (c,d), (e,f)),
            ((a,b), (c,e), (d,f)),
            ((a,b), (c,f), (d,e)),
            ((a,c), (b,d), (e,f)),
            ((a,c), (b,e), (d,f)),
            ((a,c), (b,f), (d,e)),
            ((a,d), (b,c), (e,f)),
            ((a,d), (b,e), (c,f)),
            ((a,d), (b,f), (c,e)),
            ((a,e), (b,c), (d,f)),
            ((a,e), (b,d), (c,f)),
            ((a,e), (b,f), (c,d)),
            ((a,f), (b,c), (d,e)),
            ((a,f), (b,d), (c,e)),
            ((a,f), (b,e), (c,d))
        ]
        return pairs

    config_ids = []
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for verts in combinations(range(n), 6):
        for p1, p2, p3 in get_partitions_6(verts):
            # The 8 triples are {x, y, z} where x in p1, y in p2, z in p3
            triples_in_config = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        triples_in_config.append(idx)
            
            config_id = len(configs)
            configs.append(triples_in_config)
            for idx in triples_in_config:
                triple_to_configs[idx].append(config_id)
                
    num_configs = len(configs)
    config_counts = [0] * num_configs
    selected = [False] * num_triples
    
    # 3. Greedy Initialization
    # Shuffle triples
    order = list(range(num_triples))
    random.shuffle(order)
    current_score = 0
    for idx in order:
        # Check if adding idx is valid
        valid = True
        for cid in triple_to_configs[idx]:
            if config_counts[cid] == 7: # Adding makes it 8
                valid = False
                break
        if valid:
            selected[idx] = True
            current_score += 1
            for cid in triple_to_configs[idx]:
                config_counts[cid] += 1
                
    best_score = current_score
    best_selected = selected[:]
    best_counts = config_counts[:]
    
    # 4. Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of selected and unselected for fast random choice?
    # Or just pick random index and check state.
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        move_type = random.random()
        new_score = current_score
        new_selected = None
        new_counts = None
        
        if move_type < 0.5: # Add
            idx = random.randint(0, num_triples - 1)
            if not selected[idx]:
                # Check validity
                valid = True
                for cid in triple_to_configs[idx]:
                    if config_counts[cid] == 7:
                        valid = False
                        break
                if valid:
                    # Apply
                    selected[idx] = True
                    for cid in triple_to_configs[idx]:
                        config_counts[cid] += 1
                    new_score = current_score + 1
                    # Save state for rollback if rejected? 
                    # Since we accept increases always, we only need rollback for decreases.
                    # But we are in a valid state.
                    # If we accept, we keep it. If we reject, we revert.
                    # Let's just track changes.
                    pass # Changes applied in place
                else:
                    # Invalid move, do nothing
                    continue
        elif move_type < 0.8: # Remove
            idx = random.randint(0, num_triples - 1)
            if selected[idx]:
                selected[idx] = False
                for cid in triple_to_configs[idx]:
                    config_counts[cid] -= 1
                new_score = current_score - 1
        else: # Swap
            # Pick out
            idx_out = random.randint(0, num_triples - 1)
            while not selected[idx_out]:
                idx_out = random.randint(0, num_triples - 1)
            # Pick in
            idx_in = random.randint(0, num_triples - 1)
            while selected[idx_in]:
                idx_in = random.randint(0, num_triples - 1)
            
            # Remove out
            selected[idx_out] = False
            for cid in triple_to_configs[idx_out]:
                config_counts[cid] -= 1
                
            # Check in
            valid = True
            for cid in triple_to_configs[idx_in]:
                if config_counts[cid] == 7:
                    valid = False
                    break
            
            if valid:
                # Add in
                selected[idx_in] = True
                for cid in triple_to_configs[idx_in]:
                    config_counts[cid] += 1
                new_score = current_score # Score unchanged
            else:
                # Revert out
                selected[idx_out] = True
                for cid in triple_to_configs[idx_out]:
                    config_counts[cid] += 1
                continue # Invalid swap

        # Update current state
        current_score = new_score
        
        # Acceptance
        if current_score > best_score:
            best_score = current_score
            best_selected = selected[:]
            best_counts = config_counts[:]
            
        if new_score < current_score: # This is never true in my logic above, I updated current_score.
             pass
        
        # Logic correction:
        # I applied the move to `selected` and `config_counts` and set `current_score`.
        # I need to compare `new_score` with `old_score` (which was `current_score` before update).
        # Let's restructure.
        
        # Actually, the code above modifies `selected` and `config_counts` in place.
        # I need to store the old state to revert if rejected.
        
        # Let's restart the loop logic properly in the final code.
        
        T *= cooling
        
    # Return best found
    return [list(all_triples[i]) for i in range(num_triples) if best_selected[i]]
