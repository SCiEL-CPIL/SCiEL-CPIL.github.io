import random
import time
import math
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) structures
    struct_to_triples = []
    triple_to_structs = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        return triple_to_idx[tuple(sorted((a, b, c)))]
        
    # Enumerate 6-point sets
    for six_pts in combinations(range(n), 6):
        p = list(six_pts)
        pairings = []
        def gen_pairings(remaining, current):
            if not remaining:
                pairings.append(tuple(sorted(current, key=lambda x: min(x))))
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                gen_pairings(new_remaining, current + [(first, second)])
        
        gen_pairings(p, [])
        
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_struct = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        idx = get_idx(x1, x2, x3)
                        triples_in_struct.append(idx)
            
            struct_id = len(struct_to_triples)
            struct_to_triples.append(triples_in_struct)
            for t_idx in triples_in_struct:
                triple_to_structs[t_idx].append(struct_id)
                
    num_structs = len(struct_to_triples)
    
    in_set = [False] * num_triples
    counts = [0] * num_structs
    current_size = 0
    
    def add_triple(idx):
        nonlocal current_size
        in_set[idx] = True
        current_size += 1
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] += 1
            
    def remove_triple(idx):
        nonlocal current_size
        in_set[idx] = False
        current_size -= 1
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] -= 1

    def is_valid_add(idx):
        if in_set[idx]:
            return False
        for s_idx in triple_to_structs[idx]:
            if counts[s_idx] == 7:
                return False
        return True

    # 1. Initialization: Degree-based Greedy
    # Calculate degree of each triple (number of structures it belongs to)
    degrees = [len(triple_to_structs[i]) for i in range(num_triples)]
    
    # Sort indices by degree (ascending)
    init_order = sorted(range(num_triples), key=lambda x: degrees[x])
    
    for idx in init_order:
        if is_valid_add(idx):
            add_triple(idx)
            
    best_size = current_size
    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
    
    start_time = time.time()
    time_limit = 110.0
    
    # 2. Simulated Annealing
    # T starts high, cools down
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        # Generate a move
        # Strategy: 
        # 1. Try to add a random valid triple (Improvement)
        # 2. Try to swap: remove from saturated, add random valid (Neutral/Improvement)
        # 3. Try to remove from saturated (Worse, for escape)
        
        # Find saturated structures (count == 7)
        # To be efficient, we can maintain a list or just scan. Scanning 120k is fast.
        saturated_structs = [s for s in range(num_structs) if counts[s] == 7]
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Add
            # Pick random triple not in set
            # To speed up, we can just pick random indices and check
            idx = random.randint(0, num_triples - 1)
            if not in_set[idx] and is_valid_add(idx):
                # Accept improvement
                add_triple(idx)
                if current_size > best_size:
                    best_size = current_size
                    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                T *= cooling_rate
                continue
            else:
                # Move failed or invalid, cool down slightly
                T *= cooling_rate
                continue
                
        elif move_type < 0.8 and saturated_structs:
            # Try Swap: Remove one from saturated, Add one random valid
            s_idx = random.choice(saturated_structs)
            # Pick a triple from this structure to remove
            # We need to pick one that is in_set
            # Since count is 7, 7 are in set
            t_indices = struct_to_triples[s_idx]
            in_set_in_struct = [t for t in t_indices if in_set[t]]
            if not in_set_in_struct:
                T *= cooling_rate
                continue
                
            remove_idx = random.choice(in_set_in_struct)
            
            # Find a candidate to add
            # We want to add a triple that is valid.
            # Just pick random
            add_idx = -1
            # Try a few random candidates
            for _ in range(10):
                cand = random.randint(0, num_triples - 1)
                if cand == remove_idx:
                    continue
                if not in_set[cand] and is_valid_add(cand):
                    add_idx = cand
                    break
                    
            if add_idx != -1:
                # Perform swap
                # Check if swap improves size? No, size stays same.
                # But it might allow future improvements.
                # We accept neutral moves with some probability or always?
                # In SA, neutral moves are accepted with prob 1 (or high).
                # Let's accept it.
                remove_triple(remove_idx)
                add_triple(add_idx)
                if current_size > best_size:
                    best_size = current_size
                    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                T *= cooling_rate
            else:
                T *= cooling_rate
                
        else:
            # Try Remove (Worse move) to escape local optima
            if saturated_structs:
                s_idx = random.choice(saturated_structs)
                t_indices = struct_to_triples[s_idx]
                in_set_in_struct = [t for t in t_indices if in_set[t]]
                if in_set_in_struct:
                    remove_idx = random.choice(in_set_in_struct)
                    # Accept with probability exp(-1/T) ? 
                    # Energy change is -1 (size decreases).
                    # Prob = exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        remove_triple(remove_idx)
                        if current_size > best_size:
                            best_size = current_size
                            best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                    T *= cooling_rate
            else:
                # No saturated structures? Then we might be able to add more?
                # If no saturated, we should have been able to add in the 'Add' phase.
                # Unless we missed some.
                T *= cooling_rate

        # Reset temperature if we are stuck? 
        # Or just let it cool.
        
    return best_solution
