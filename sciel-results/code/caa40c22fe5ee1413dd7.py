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

    # 1. Algebraic Initialization: Lines of AG(4,2)
    # Points are 0..15. Map to (x,y) in Z4xZ4 or F2^4.
    # Using F2^4 representation: 0..15 are 4-bit integers.
    # A line is {u, v, u^v} for u != v, u != 0, v != 0?
    # In AG(4,2), lines are sets {a, b, a+b} where a,b are distinct non-zero vectors?
    # No, in affine space, lines are {p, p+u, p+2u} but in char 2, p+2u = p.
    # So lines in AG(4,2) are {p, q, p+q} where p,q are distinct points.
    # Wait, if p,q are points in F2^4, then p+q is the third point.
    # The set of all such triples {p, q, p^q} for all distinct p,q.
    # Note that {p, q, p^q} is the same as {p, p^q, q} etc.
    # Number of such triples:
    # Choose 2 distinct points p, q. There are 16*15 = 240 ordered pairs.
    # Each triple {a,b,c} with a^b^c=0 is generated 6 times (3 choices for "sum", 2 orders for others? No, 3 choices for which one is the sum of the other two, and 2! order of the addends).
    # So 240 / 6 = 40 triples.
    
    # Generate these 40 triples
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            k = i ^ j
            if k == i or k == j:
                continue # i^j is 0 only if i==j, which is handled by i!=j. 
                         # But i^j could be i? No, j=0. 
                         # If j=0, k=i. Then {i, 0, i}. Distinct? No, i==k.
                         # So we need distinct points.
                         # In F2, i^j=0 iff i=j.
                         # i^j=i iff j=0.
                         # So if 0 is involved, say i=0, j=1, k=1. {0,1,1} not distinct.
                         # So lines must not contain 0 as the "sum" of two identicals?
                         # Actually, in AG(4,2), the points are the vectors.
                         # A line is {u, v, u+v}.
                         # If u=0, v!=0, then {0, v, v}. Not a valid triple of distinct points.
                         # So 0 is not part of any line in this definition?
                         # Wait. In affine geometry, a line has 2 points? No, in AG(n,2), a line has 2 points?
                         # No, in AG(n,2), the smallest line has 2 points?
                         # Definition: A line in AG(n,q) has q points. For q=2, it has 2 points.
                         # But we need triples.
                         # So AG(4,2) lines are pairs? No, that doesn't help.
                         # We need 3-uniform hypergraph.
                         # Maybe use AG(4,3)? No, 16 points.
                         # How about using the 40 "lines" of the 4-dimensional affine geometry over F2 is not standard for 3-uniform.
                         # Standard 3-uniform Steiner systems exist for v=7, 13, ... not 16.
                         
                         # Let's stick to Random Initialization but with a "Bias" towards algebraic structures?
                         # Or just use the Random Initialization from the seed but improve the Search.
                         # The seed's search is weak.
                         
                         # Let's try a different algebraic structure: 
                         # The 16 points can be vertices of a 4-cube.
                         # Triples can be faces? No.
                         
                         # Let's go back to Random Initialization but optimize the Local Search.
                         # The main bottleneck in the seed is the `random.shuffle(order)` in the inner loop.
                         # We can maintain a list of "candidates" or just iterate 0..N-1.
                         # Also, the kick strategy is random.
                         
                         # New Strategy:
                         # 1. Random Init (as in seed).
                         # 2. Greedy Add: Iterate 0..N-1, add if valid.
                         # 3. Local Search:
                         #    - Find all structures with count 7.
                         #    - If none, we are at a local max.
                         #    - Pick a random structure with count 7.
                         #    - Remove a random triple from it.
                         #    - Repeat Greedy Add.
                         
    random.seed(42)
    # 1. Greedy Initialization
    # Start with empty set? Or random?
    # Random start is often better to avoid bias.
    order = list(range(num_triples))
    random.shuffle(order)
    for idx in order:
        if is_valid_add(idx):
            add_triple(idx)
            
    best_size = current_size
    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
    
    start_time = time.time()
    time_limit = 110.0
    
    # 2. Local Search with Targeted Kicks
    while time.time() - start_time < time_limit:
        # Greedy Add Pass
        # We iterate through all triples and add any that are valid.
        # To make it efficient, we just loop 0 to N-1.
        # We can shuffle the order of iteration to avoid bias.
        iter_order = list(range(num_triples))
        random.shuffle(iter_order)
        
        added_any = False
        for idx in iter_order:
            if not in_set[idx] and is_valid_add(idx):
                add_triple(idx)
                added_any = True
                if current_size > best_size:
                    best_size = current_size
                    best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
                    
        if not added_any:
            # Stuck. Need to kick.
            if time.time() - start_time > time_limit - 5:
                break
                
            # Find structures with count 7
            # This scan is O(N_structs). N_structs ~ 120k. This is fast.
            candidates = [s_idx for s_idx in range(num_structs) if counts[s_idx] == 7]
            
            if not candidates:
                # This shouldn't happen if we are stuck and can't add anything.
                # If we can't add anything, every missing triple completes a structure.
                # That structure must have count 7.
                # So candidates should not be empty.
                # Unless we are in a weird state.
                # If empty, break or random kick.
                random_idx = random.randint(0, num_triples-1)
                if in_set[random_idx]:
                    remove_triple(random_idx)
                continue
            
            # Pick a random candidate structure
            s_idx = random.choice(candidates)
            
            # Remove a random triple from this structure
            # We need to pick a triple from struct_to_triples[s_idx] that is currently in_set.
            # Since count is 7, 7 of the 8 are in set.
            t_indices = struct_to_triples[s_idx]
            in_set_in_struct = [t for t in t_indices if in_set[t]]
            
            if in_set_in_struct:
                remove_idx = random.choice(in_set_in_struct)
                remove_triple(remove_idx)
                
    return best_solution
