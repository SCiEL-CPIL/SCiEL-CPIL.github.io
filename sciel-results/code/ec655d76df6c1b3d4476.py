import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    m = len(all_triples)  # 560 triples
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # For each triple, store which K(2,2,2)s it belongs to (as indices)
    # And for each K(2,2,2), store its 8 triple indices
    
    k222_triples = []  # list of sets of 8 triple indices
    triple_to_k222s = [[] for _ in range(m)]
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append(tuple(sorted((x, y, z))))
            
            trans_indices = tuple(triple_to_idx[t] for t in transversals)
            k_idx = len(k222_triples)
            k222_triples.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_triples)
    
    # Build conflict structure: for each triple, which other triples conflict with it
    # Two triples conflict if they are in the same K(2,2,2) and that K(2,2,2) has 6 other triples
    # Actually, we need: if we add triple t, it conflicts if there exists a K(2,2,2) where
    # all 7 other triples are already present.
    
    # For efficiency, precompute for each triple a list of (k_idx, other_triples_in_k)
    # where other_triples_in_k is a frozenset of the 7 other triples in that K(2,2,2)
    
    triple_conflict_info = []
    for t_idx in range(m):
        conflicts = []
        for k_idx in triple_to_k222s[t_idx]:
            others = tuple(k222_triples[k_idx])
            others_set = frozenset(others) - frozenset([t_idx])
            conflicts.append((k_idx, others_set))
        triple_conflict_info.append(conflicts)
    
    # Greedy construction with multiple strategies
    def greedy_construct(order, use_weights=False):
        present = np.zeros(m, dtype=np.uint8)
        k222_count = np.zeros(num_k222s, dtype=np.uint8)
        current_set = []
        
        for t_idx in order:
            # Check if adding this triple would complete any K(2,2,2)
            can_add = True
            for k_idx in triple_to_k222s[t_idx]:
                if k222_count[k_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                present[t_idx] = 1
                current_set.append(t_idx)
                for k_idx in triple_to_k222s[t_idx]:
                    k222_count[k_idx] += 1
        
        return current_set, present, k222_count
    
    # Local search: try to add more triples by removing conflicting ones
    def local_search(initial_set, present, k222_count, max_time):
        current_set = list(initial_set)
        current_count = len(current_set)
        best_set = list(current_set)
        best_count = current_count
        best_present = present.copy()
        best_k222_count = k222_count.copy()
        
        # Precompute for each triple, the list of K(2,2,2)s it's in
        # For removal, we need to know which K(2,2,2)s would be freed
        
        # Strategy: try adding each absent triple, and if it conflicts,
        # try removing one of the conflicting triples to free up space
        
        absent_triples = [i for i in range(m) if not present[i]]
        random.shuffle(absent_triples)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            for add_idx in absent_triples:
                if time.time() > max_time:
                    break
                
                # Check conflicts
                conflicting_k222s = []
                for k_idx in triple_to_k222s[add_idx]:
                    if k222_count[k_idx] == 7:
                        conflicting_k222s.append(k_idx)
                
                if not conflicting_k222s:
                    # Can add directly
                    present[add_idx] = 1
                    current_set.append(add_idx)
                    for k_idx in triple_to_k222s[add_idx]:
                        k222_count[k_idx] += 1
                    current_count += 1
                    absent_triples.remove(add_idx)
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_set = list(current_set)
                        best_present = present.copy()
                        best_k222_count = k222_count.copy()
                    continue
                
                # Try to resolve conflicts by removing one triple from each conflicting K(2,2,2)
                # For each conflicting K(2,2,2), find a triple to remove
                # We want to remove a triple that's in the fewest K(2,2,2)s to minimize disruption
                
                for k_idx in conflicting_k222s:
                    if time.time() > max_time:
                        break
                    
                    # Get the 7 triples in this K(2,2,2) that are present
                    k_triples = k222_triples[k_idx]
                    present_in_k = [t for t in k_triples if present[t] and t != add_idx]
                    
                    if len(present_in_k) != 7:
                        continue
                    
                    # Find the triple to remove: one that's in the fewest other K(2,2,2)s
                    # (i.e., removing it frees up the least other K(2,2,2)s)
                    min_remove = None
                    min_count = float('inf')
                    for t in present_in_k:
                        # Count how many K(2,2,2)s have this triple and are at count 7
                        conflict_count = 0
                        for k2 in triple_to_k222s[t]:
                            if k222_count[k2] == 7:
                                conflict_count += 1
                        if conflict_count < min_count:
                            min_count = conflict_count
                            min_remove = t
                    
                    if min_remove is None:
                        continue
                    
                    # Remove min_remove and add add_idx
                    # Check if this is net positive: we remove 1, add 1, but might free up other adds
                    # Only do this if min_remove is in fewer conflicts than add_idx would create
                    add_conflicts = len(conflicting_k222s)
                    
                    if min_count <= add_conflicts:
                        # Remove min_remove
                        present[min_remove] = 0
                        current_set.remove(min_remove)
                        for k2 in triple_to_k222s[min_remove]:
                            k222_count[k2] -= 1
                        
                        # Add add_idx
                        present[add_idx] = 1
                        current_set.append(add_idx)
                        for k2 in triple_to_k222s[add_idx]:
                            k222_count[k2] += 1
                        
                        absent_triples.remove(add_idx)
                        improved = True
                        
                        if current_count > best_count:
                            best_count = current_count
                            best_set = list(current_set)
                            best_present = present.copy()
                            best_k222_count = k222_count.copy()
        
        return best_set, best_present, best_k222_count, best_count
    
    # Simulated annealing
    def simulated_annealing(initial_set, present, k222_count, max_time, T0=1.0, T_min=0.01, cooling=0.9999):
        current_set = list(initial_set)
        current_count = len(current_set)
        best_set = list(current_set)
        best_count = current_count
        best_present = present.copy()
        best_k222_count = k222_count.copy()
        
        T = T0
        step = 0
        no_improve_steps = 0
        
        while time.time() < max_time and T > T_min:
            step += 1
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random absent triple
                absent = [i for i in range(m) if not present[i]]
                if not absent:
                    T *= cooling
                    continue
                add_idx = random.choice(absent)
                
                # Check if can add
                can_add = True
                for k_idx in triple_to_k222s[add_idx]:
                    if k222_count[k_idx] == 7:
                        can_add = False
                        break
                
                if can_add:
                    # Always accept an improving move
                    present[add_idx] = 1
                    current_set.append(add_idx)
                    for k_idx in triple_to_k222s[add_idx]:
                        k222_count[k_idx] += 1
                    current_count += 1
                    no_improve_steps = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_set = list(current_set)
                        best_present = present.copy()
                        best_k222_count = k222_count.copy()
                else:
                    # Can't add, skip
                    pass
            
            elif move_type < 0.7:
                # Try to remove a random present triple and add a different one
                if not current_set:
                    T *= cooling
                    continue
                
                remove_idx = random.choice(current_set)
                
                # Find a triple to add that was conflicting with remove_idx
                # Look at K(2,2,2)s that remove_idx is in
                candidate_adds = []
                for k_idx in triple_to_k222s[remove_idx]:
                    for t in k222_triples[k_idx]:
                        if t != remove_idx and not present[t]:
                            candidate_adds.append(t)
                
                if candidate_adds:
                    add_idx = random.choice(candidate_adds)
                    
                    # Remove remove_idx
                    present[remove_idx] = 0
                    current_set.remove(remove_idx)
                    for k_idx in triple_to_k222s[remove_idx]:
                        k222_count[k_idx] -= 1
                    
                    # Try to add add_idx
                    can_add = True
                    for k_idx in triple_to_k222s[add_idx]:
                        if k222_count[k_idx] == 7:
                            can_add = False
                            break
                    
                    if can_add:
                        present[add_idx] = 1
                        current_set.append(add_idx)
                        for k_idx in triple_to_k222s[add_idx]:
                            k222_count[k_idx] += 1
                        # Count stays the same
                    else:
                        # Revert: put remove_idx back
                        present[remove_idx] = 1
                        current_set.append(remove_idx)
                        for k_idx in triple_to_k222s[remove_idx]:
                            k222_count[k_idx] += 1
                        # Reject
                        if random.random() > np.exp(0 / T):  # always reject neutral
                            pass
                        else:
                            pass
                else:
                    # Just remove
                    present[remove_idx] = 0
                    current_set.remove(remove_idx)
                    for k_idx in triple_to_k222s[remove_idx]:
                        k222_count[k_idx] -= 1
                    current_count -= 1
                    
                    # Accept with probability exp(-1/T)
                    if random.random() < np.exp(-1.0 / T):
                        no_improve_steps += 1
                    else:
                        # Revert
                        present[remove_idx] = 1
                        current_set.append(remove_idx)
                        for k_idx in triple_to_k222s[remove_idx]:
                            k222_count[k_idx] += 1
                        current_count += 1
            
            else:
                # Random perturbation: remove a few random triples
                num_remove = random.randint(1, 3)
                to_remove = random.sample(current_set, min(num_remove, len(current_set)))
                for t in to_remove:
                    present[t] = 0
                    current_set.remove(t)
                    for k_idx in triple_to_k222s[t]:
                        k222_count[k_idx] -= 1
                current_count -= len(to_remove)
                no_improve_steps += 1
            
            T *= cooling
            
            if current_count > best_count:
                best_count = current_count
                best_set = list(current_set)
                best_present = present.copy()
                best_k222_count = k222_count.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # If no improvement for too long, restart from best
            if no_improve_steps > 5000:
                break
        
        return best_set, best_present, best_k222_count, best_count
    
    # Main search loop
    best_result = []
    best_count = 0
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_greedy = 15
    for g in range(num_greedy):
        if time.time() > deadline - 20:
            break
        
        order = list(range(m))
        random.shuffle(order)
        
        current_set, present, k222_count = greedy_construct(order)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_result = [list(all_triples[i]) for i in current_set]
    
    # Phase 2: Local search on best greedy result
    if time.time() < deadline - 30:
        # Reconstruct present and k222_count for best_result
        present = np.zeros(m, dtype=np.uint8)
        k222_count = np.zeros(num_k222s, dtype=np.uint8)
        for t in best_result:
            t_idx = triple_to_idx[tuple(t)]
            present[t_idx] = 1
            for k_idx in triple_to_k222s[t_idx]:
                k222_count[k_idx] += 1
        
        best_set, best_present, best_k222_count, best_count = local_search(
            best_result, present, k222_count, deadline - 15
        )
        best_result = [list(all_triples[i]) for i in best_set]
    
    # Phase 3: Simulated annealing with multiple restarts
    if time.time() < deadline - 10:
        num_sa_restarts = 5
        for sa in range(num_sa_restarts):
            if time.time() > deadline - 5:
                break
            
            # Start from current best with some perturbation
            if best_result:
                start_set = list(best_result)
                # Perturb: remove 5-15 random triples
                num_perturb = random.randint(5, min(15, len(start_set)))
                if start_set:
                    to_remove = random.sample(start_set, num_perturb)
                    start_set = [t for t in start_set if t not in set(to_remove)]
            else:
                start_set = []
            
            present = np.zeros(m, dtype=np.uint8)
            k222_count = np.zeros(num_k222s, dtype=np.uint8)
            for t in start_set:
                t_idx = triple_to_idx[tuple(t)]
                present[t_idx] = 1
                for k_idx in triple_to_k222s[t_idx]:
                    k222_count[k_idx] += 1
            
            sa_set, sa_present, sa_k222_count, sa_count = simulated_annealing(
                start_set, present, k222_count, deadline - 5,
                T0=1.0, T_min=0.01, cooling=0.99995
            )
            
            if sa_count > best_count:
                best_count = sa_count
                best_result = [list(all_triples[i]) for i in sa_set]
            
            # Also try local search on the SA result
            if time.time() < deadline - 3:
                ls_set, ls_present, ls_k222_count, ls_count = local_search(
                    sa_set, sa_present, sa_k222_count, deadline - 2
                )
                if ls_count > best_count:
                    best_count = ls_count
                    best_result = [list(all_triples[i]) for i in ls_set]
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 1 and best_result:
        present = np.zeros(m, dtype=np.uint8)
        k222_count = np.zeros(num_k222s, dtype=np.uint8)
        for t in best_result:
            t_idx = triple_to_idx[tuple(t)]
            present[t_idx] = 1
            for k_idx in triple_to_k222s[t_idx]:
                k222_count[k_idx] += 1
        
        final_set, final_present, final_k222_count, final_count = local_search(
            best_result, present, k222_count, deadline - 0.5
        )
        if final_count > best_count:
            best_count = final_count
            best_result = [list(all_triples[i]) for i in final_set]
    
    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
