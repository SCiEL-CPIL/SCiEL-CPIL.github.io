import random
import time
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
    best_size = 0
    best_solution = []
    
    start_time = time.time()
    time_limit = 115.0

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

    def try_add_all():
        nonlocal current_size, best_size, best_solution
        added_any = True
        while added_any:
            added_any = False
            # Random order to break symmetry
            candidates = [i for i in range(num_triples) if not in_set[i]]
            random.shuffle(candidates)
            for idx in candidates:
                if is_valid_add(idx):
                    add_triple(idx)
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                    break

    def try_1_out_2_in():
        nonlocal current_size, best_size, best_solution
        # Try to remove one and add two
        current_indices = [i for i in range(num_triples) if in_set[i]]
        random.shuffle(current_indices)
        
        for u in current_indices:
            if time.time() - start_time > time_limit - 5:
                return False
            
            # Remove u
            remove_triple(u)
            
            # Find candidates to add
            candidates = []
            for v in range(num_triples):
                if not in_set[v] and is_valid_add(v):
                    candidates.append(v)
            
            if len(candidates) >= 2:
                # Try to find two compatible candidates
                # Randomly shuffle candidates to explore different pairs
                random.shuffle(candidates)
                for i in range(len(candidates)):
                    v1 = candidates[i]
                    # Tentatively add v1
                    add_triple(v1)
                    for j in range(i+1, len(candidates)):
                        v2 = candidates[j]
                        if is_valid_add(v2):
                            add_triple(v2)
                            # Success: size increased by 1
                            if current_size > best_size:
                                best_size = current_size
                                best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                            return True
                        # If v2 not valid, we don't remove v1 yet, just check next v2?
                        # No, v1 might block v2. We need to check if v1 and v2 are mutually exclusive.
                        # If is_valid_add(v2) is False, it means v2 is blocked by current state (which includes v1).
                        # So we just try next v2.
                    # If no v2 found compatible with v1, remove v1
                    remove_triple(v1)
            
            # Restore u
            add_triple(u)
            
        return False

    # Main Loop
    random.seed(42)
    
    # Multiple restarts
    num_restarts = 10
    for _ in range(num_restarts):
        # Reset state
        in_set = [False] * num_triples
        counts = [0] * num_structs
        current_size = 0
        
        # Greedy Init
        order = list(range(num_triples))
        random.shuffle(order)
        for idx in order:
            if is_valid_add(idx):
                add_triple(idx)
        
        if current_size > best_size:
            best_size = current_size
            best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
        
        # Local Search
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # 1. Try to add
            try_add_all()
            
            # 2. If stuck, try 1-out-2-in
            if try_1_out_2_in():
                improved = True
                # After a successful 1-out-2-in, try to add more
                try_add_all()
                
    return best_solution
