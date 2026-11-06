import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    n_all = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflicts: for each triple, find all 6-point configurations where it's one of the 8 transversals
    # For each triple T = (x,y,z), find all (x',y',z') disjoint from {x,y,z}
    # The 8 transversals are all combinations of choosing one from {x,x'}, one from {y,y'}, one from {z,z'}
    # T itself is (x,y,z). The other 7 are the rest.
    # Two triples conflict if they are in the same K(2,2,2) configuration.
    
    # Build conflict graph: conflict[i][j] = True if triples i and j are in some common K(2,2,2)
    # Actually, we need: for each triple, which OTHER triples block it (i.e., form a K(2,2,2) with it)
    
    # For efficiency, precompute for each triple the list of "partner" triples that form K(2,2,2) with it
    # A K(2,2,2) is defined by 6 points: {x,x'}, {y,y'}, {z,z'}
    # The 8 transversals are all (a,b,c) where a in {x,x'}, b in {y,y'}, c in {z,z'}
    
    # For a given triple (x,y,z), the partners are the other 7 transversals for some choice of x',y',z'
    # Multiple (x',y',z') choices may yield overlapping partner sets.
    
    # Let's precompute: for each triple index, the set of triple indices that conflict with it
    # Two triples conflict if there exist 6 points forming a K(2,2,2) containing both triples as transversals.
    
    # This is expensive to precompute fully. Instead, let's use a faster on-the-fly check.
    
    # Represent the system as a set of triple indices and use bitmasks for fast membership.
    # pair_masks[a][b] is a bitmask where bit c is set if (a,b,c) is in the system (with a<b<c)
    
    def check_violation_fast(triple, pair_masks):
        """
        Returns None if triple can be added, or returns a list of blocking triples (as tuples) if it conflicts.
        """
        x, y, z = triple
        # Need to find x', y', z' such that all 8 transversals are present.
        # Transversals: (x or x', y or y', z or z')
        # We check if (x',y,z), (x,y',z), (x',y',z) are present first (3 checks)
        # Then for each z', check (x,y,z'), (x',y,z'), (x,y',z'), (x',y',z')
        
        for x_prime in range(16):
            if x_prime in (x, y, z):
                continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime):
                    continue
                # Check 3 transversals with z fixed
                a1, b1, c1 = sorted((x_prime, y, z))
                if not (pair_masks[a1][b1] >> c1) & 1:
                    continue
                a2, b2, c2 = sorted((x, y_prime, z))
                if not (pair_masks[a2][b2] >> c2) & 1:
                    continue
                a3, b3, c3 = sorted((x_prime, y_prime, z))
                if not (pair_masks[a3][b3] >> c3) & 1:
                    continue
                
                # Now check z_prime
                for z_prime in range(16):
                    if z_prime in (x, y, z, x_prime, y_prime):
                        continue
                    # Check remaining 4 transversals
                    a4, b4, c4 = sorted((x, y, z_prime))
                    if not (pair_masks[a4][b4] >> c4) & 1:
                        continue
                    a5, b5, c5 = sorted((x_prime, y, z_prime))
                    if not (pair_masks[a5][b5] >> c5) & 1:
                        continue
                    a6, b6, c6 = sorted((x, y_prime, z_prime))
                    if not (pair_masks[a6][b6] >> c6) & 1:
                        continue
                    a7, b7, c7 = sorted((x_prime, y_prime, z_prime))
                    if not (pair_masks[a7][b7] >> c7) & 1:
                        continue
                    
                    # Found a conflict. Return the 7 other triples.
                    return [
                        tuple(sorted((x_prime, y, z))),
                        tuple(sorted((x, y_prime, z))),
                        tuple(sorted((x_prime, y_prime, z))),
                        tuple(sorted((x, y, z_prime))),
                        tuple(sorted((x_prime, y, z_prime))),
                        tuple(sorted((x, y_prime, z_prime))),
                        tuple(sorted((x_prime, y_prime, z_prime)))
                    ]
        return None

    def add_triple(triple, system_set, pair_masks):
        x, y, z = triple
        a, b, c = sorted((x, y, z))
        pair_masks[a][b] |= (1 << c)
        system_set.add(tuple(sorted((x, y, z))))

    def remove_triple(triple, system_set, pair_masks):
        x, y, z = triple
        a, b, c = sorted((x, y, z))
        pair_masks[a][b] &= ~(1 << c)
        system_set.remove(tuple(sorted((x, y, z))))

    # Phase 1: Greedy Initialization with multiple random orders
    best_system_set = None
    best_pair_masks = None
    best_size = 0
    
    for attempt in range(15):
        if time.time() > deadline - 40:
            break  # Leave time for local search
        
        pair_masks = [[0] * 16 for _ in range(16)]
        system_set = set()
        
        order = list(all_triples)
        random.shuffle(order)
        
        for t in order:
            if check_violation_fast(t, pair_masks) is None:
                add_triple(t, system_set, pair_masks)
        
        if len(system_set) > best_size:
            best_size = len(system_set)
            best_system_set = set(system_set)
            best_pair_masks = [row[:] for row in pair_masks]

    if best_system_set is None:
        # Fallback star
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

    # Phase 2: Local Search with smart kicking
    current_set = set(best_system_set)
    current_masks = [row[:] for row in best_pair_masks]
    current_size = len(current_set)
    
    # Build list of triples not in current set
    remaining_triples = [t for t in all_triples if t not in current_set]
    random.shuffle(remaining_triples)
    remaining_idx = 0
    
    # Track best ever
    global_best_size = best_size
    global_best_set = set(best_system_set)
    global_best_masks = [row[:] for row in best_pair_masks]
    
    # Local search: try to add triples, removing blockers if needed
    # Use a temperature-like parameter that decreases over time
    step_count = 0
    
    while time.time() < deadline - 5:
        if remaining_idx >= len(remaining_triples):
            # Reset remaining list
            remaining_triples = [t for t in all_triples if t not in current_set]
            if not remaining_triples:
                break
            random.shuffle(remaining_triples)
            remaining_idx = 0
            step_count = 0
        
        t = remaining_triples[remaining_idx]
        remaining_idx += 1
        step_count += 1
        
        conflict_list = check_violation_fast(t, current_masks)
        
        if conflict_list is None:
            # Can add directly
            add_triple(t, current_set, current_masks)
            current_size += 1
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_set = set(current_set)
                global_best_masks = [row[:] for row in current_masks]
        else:
            # Conflict exists. Try to kick out a blocker.
            # Choose blocker strategically: prefer blockers that are "less useful" 
            # (i.e., have fewer conflicts with other triples)
            
            # For simplicity, pick a random blocker but with some bias
            blocker = random.choice(conflict_list)
            
            if blocker in current_set:
                # Remove the blocker
                remove_triple(blocker, current_set, current_masks)
                current_size -= 1
                
                # Try to add t
                if check_violation_fast(t, current_masks) is None:
                    add_triple(t, current_set, current_masks)
                    current_size += 1
                    if current_size > global_best_size:
                        global_best_size = current_size
                        global_best_set = set(current_set)
                        global_best_masks = [row[:] for row in current_masks]
                else:
                    # Still conflicts. Try removing one more blocker if possible.
                    conflict_list2 = check_violation_fast(t, current_masks)
                    if conflict_list2:
                        blocker2 = random.choice(conflict_list2)
                        if blocker2 in current_set and blocker2 != t:
                            remove_triple(blocker2, current_set, current_masks)
                            current_size -= 1
                            
                            if check_violation_fast(t, current_masks) is None:
                                add_triple(t, current_set, current_masks)
                                current_size += 1
                                if current_size > global_best_size:
                                    global_best_size = current_size
                                    global_best_set = set(current_set)
                                    global_best_masks = [row[:] for row in current_masks]
                            else:
                                # Revert both removals
                                add_triple(blocker2, current_set, current_masks)
                                current_size += 1
                    # Revert the first removal if we didn't succeed
                    if t not in current_set:
                        add_triple(blocker, current_set, current_masks)
                        current_size += 1
    
    # Return the best found
    return [list(t) for t in global_best_set]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
