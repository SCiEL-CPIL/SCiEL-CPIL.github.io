import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) structures
    # A structure is 3 disjoint pairs.
    # We can represent a structure by the 6 points involved, sorted, and the pairing.
    # But for counting, we just need a unique ID for each set of 3 pairs.
    # Let's enumerate all ways to pick 3 disjoint pairs.
    structs = []
    # To avoid duplicates, we can enforce an ordering on the pairs and the points within pairs.
    # Pair 1: (p1, p2) with p1 < p2
    # Pair 2: (p3, p4) with p3 < p4 and p3 > p2? No, just disjoint.
    # To ensure unique representation of the *set* of 3 pairs:
    # Sort the pairs by their min element, then max element.
    # Within a pair, sort elements.
    
    # This enumeration might be tricky to do uniquely without duplicates.
    # Alternative: Iterate over all combinations of 6 points.
    # For each 6 points, there are 15 ways to partition into 3 pairs.
    # Total 6-point sets: C(16,6) = 8008.
    # 8008 * 15 = 120,120 structures. This matches.
    
    # Let's build the mapping.
    # struct_id -> list of 8 triple indices
    # triple_idx -> list of struct_ids
    
    struct_to_triples = []
    triple_to_structs = [[] for _ in range(num_triples)]
    
    # Helper to get triple index
    def get_idx(a, b, c):
        return triple_to_idx[tuple(sorted((a, b, c)))]
    
    # Enumerate 6-point sets
    for six_pts in combinations(range(n), 6):
        # Generate all 15 pairings of these 6 points
        # This is a known combinatorial object: number of perfect matchings in K6 is 15.
        # We can hardcode the 15 pairings for 6 items 0..5.
        # Or generate them.
        # Let's generate them.
        p = list(six_pts)
        # Recursive generation of pairings
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
            # pairing is a tuple of 3 pairs, e.g., ((a,b), (c,d), (e,f))
            # The 8 transversal triples are formed by picking one from each pair.
            p1, p2, p3 = pairing
            triples_in_struct = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        idx = get_idx(x1, x2, x3)
                        triples_in_struct.append(idx)
            
            # Add to structures
            struct_id = len(struct_to_triples)
            struct_to_triples.append(triples_in_struct)
            for t_idx in triples_in_struct:
                triple_to_structs[t_idx].append(struct_id)
                
    num_structs = len(struct_to_triples)
    
    # State
    in_set = [False] * num_triples
    counts = [0] * num_structs
    current_size = 0
    best_size = 0
    best_solution = []
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    def add_triple(idx):
        nonlocal current_size
        # Check validity
        for s_idx in triple_to_structs[idx]:
            if counts[s_idx] == 7:
                return False
        # Add
        in_set[idx] = True
        current_size += 1
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] += 1
        return True
        
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

    # 1. Greedy Initialization
    random.seed(42)
    order = list(range(num_triples))
    random.shuffle(order)
    for idx in order:
        if is_valid_add(idx):
            add_triple(idx)
            
    best_size = current_size
    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
    
    # 2. Local Search with Kicks
    # We try to add any triple. If we can't, we kick.
    # To speed up "try to add any triple", we can maintain a list of candidates?
    # Or just shuffle and try.
    
    while time.time() - start_time < time_limit:
        # Try to add as many as possible
        added_any = True
        while added_any and (time.time() - start_time < time_limit):
            added_any = False
            # Shuffle indices to avoid bias
            random.shuffle(order)
            for idx in order:
                if not in_set[idx] and is_valid_add(idx):
                    add_triple(idx)
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                    break # Restart search from beginning after one add? 
                         # Or continue? Continuing is faster.
                         # Let's continue to add more if possible.
                         # But `order` is fixed in this inner loop iteration.
                         # It's better to just scan all `order` and add all valid ones.
                         # Let's modify: don't break, just add all valid ones in one pass.
                         # But if we add one, it might invalidate others or enable none.
                         # So a full scan is good.
        
        # If we added nothing in the last pass, we are stuck.
        # We need to kick.
        if not added_any:
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
                
            # Kick: Remove k random triples
            k = random.randint(2, 5)
            current_indices = [i for i in range(num_triples) if in_set[i]]
            if len(current_indices) < k:
                continue
            to_remove = random.sample(current_indices, k)
            for idx in to_remove:
                remove_triple(idx)
                
    return best_solution
