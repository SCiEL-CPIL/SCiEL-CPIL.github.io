import time
import random
import sys
from itertools import combinations, product

def _precompute(n=16):
    """Precompute all triples and K(2,2,2) configurations."""
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Build K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs (a,a'), (b,b'), (c,c')
    # and consists of the 8 triples choosing one from each pair
    triple_to_k222s = [[] for _ in range(n_triples)]
    k222_list = []
    
    # Generate all sets of 6 distinct points
    for points in combinations(range(n), 6):
        # Generate all perfect matchings (partitions into 3 pairs)
        # For 6 points, there are 15 perfect matchings
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # matching = [(a,a'), (b,b'), (c,c')]
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = set()
            for t in k_triples:
                k_indices.add(triple_to_idx[t])
            
            k222_idx = len(k222_list)
            k222_list.append(frozenset(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
    
    return all_triples, triple_to_k222s, k222_list

def _check_conflict(triple_idx, present, triple_to_k222s, k222_list):
    """Check if adding triple_idx would create a K(2,2,2)."""
    for k_idx in triple_to_k222s[triple_idx]:
        k_set = k222_list[k_idx]
        # All other 7 triples in this K must be present
        if all(other in present for other in k_set if other != triple_idx):
            return True
    return False

def _find_conflicts(triple_idx, present, triple_to_k222s, k222_list):
    """Find which K(2,2,2) configurations would be completed."""
    conflicts = []
    for k_idx in triple_to_k222s[triple_idx]:
        k_set = k222_list[k_idx]
        if all(other in present for other in k_set if other != triple_idx):
            conflicts.append(k_idx)
    return conflicts

def _greedy_init(n_triples, triple_to_k222s, k222_list, deadline):
    """Greedy initialization with random ordering."""
    order = list(range(n_triples))
    random.shuffle(order)
    present = set()
    for idx in order:
        if time.time() > deadline:
            break
        if not _check_conflict(idx, present, triple_to_k222s, k222_list):
            present.add(idx)
    return present

def _local_search(present, all_indices, triple_to_k222s, k222_list, deadline, max_iters=10000):
    """
    Local search with add and swap moves.
    """
    best_set = set(present)
    best_size = len(present)
    iters = 0
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        
        # Try to add a random absent triple
        u = random.choice(all_indices)
        if u in present:
            continue
        
        conflicts = _find_conflicts(u, present, triple_to_k222s, k222_list)
        
        if not conflicts:
            # Can add directly
            present.add(u)
            no_improve = 0
            if len(present) > best_size:
                best_size = len(present)
                best_set = set(present)
        else:
            # Try swap: remove one conflicting triple, add u
            # Pick a random conflict
            k_idx = random.choice(conflicts)
            k_set = k222_list[k_idx]
            
            # Candidates to remove: triples in k_set that are in present (excluding u)
            candidates = [t for t in k_set if t != u and t in present]
            
            if candidates:
                rem = random.choice(candidates)
                present.remove(rem)
                present.add(u)
                
                # Check if u is now valid
                if not _check_conflict(u, present, triple_to_k222s, k222_list):
                    no_improve = 0
                    if len(present) > best_size:
                        best_size = len(present)
                        best_set = set(present)
                else:
                    # Revert
                    present.remove(u)
                    present.add(rem)
                    no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_set

def _simulated_annealing(present, all_indices, triple_to_k222s, k222_list, deadline, 
                          T_start=1.0, T_end=0.01, cooling=0.999):
    """
    Simulated annealing with add and swap moves.
    """
    best_set = set(present)
    best_size = len(present)
    T = T_start
    iters = 0
    max_iters = 50000
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        T *= cooling
        if T < T_end:
            T = T_end
        
        # Try to add a random absent triple
        u = random.choice(all_indices)
        if u in present:
            continue
        
        conflicts = _find_conflicts(u, present, triple_to_k222s, k222_list)
        
        if not conflicts:
            # Can add directly - always accept
            present.add(u)
            if len(present) > best_size:
                best_size = len(present)
                best_set = set(present)
        else:
            # Try swap
            k_idx = random.choice(conflicts)
            k_set = k222_list[k_idx]
            candidates = [t for t in k_set if t != u and t in present]
            
            if candidates:
                rem = random.choice(candidates)
                present.remove(rem)
                present.add(u)
                
                # Check validity
                if not _check_conflict(u, present, triple_to_k222s, k222_list):
                    if len(present) > best_size:
                        best_size = len(present)
                        best_set = set(present)
                else:
                    # Invalid - accept with probability based on temperature
                    if random.random() > T:
                        # Revert
                        present.remove(u)
                        present.add(rem)
    
    return best_set

def _perturb(present, all_indices, triple_to_k222s, k222_list, n_remove=3, n_add=3):
    """Perturb the solution by removing and adding random triples."""
    perturbed = set(present)
    
    # Remove random elements
    for _ in range(n_remove):
        if perturbed:
            perturbed.remove(random.choice(list(perturbed)))
    
    # Add random valid elements
    attempts = 0
    added = 0
    while added < n_add and attempts < 50:
        attempts += 1
        u = random.choice(all_indices)
        if u not in perturbed and not _check_conflict(u, perturbed, triple_to_k222s, k222_list):
            perturbed.add(u)
            added += 1
    
    return perturbed

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute(n)
    n_triples = len(all_triples)
    all_indices = list(range(n_triples))
    
    # Initial solution via greedy
    current_set = _greedy_init(n_triples, triple_to_k222s, k222_list, deadline)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Multiple rounds of local search and simulated annealing
    round_num = 0
    while time.time() < deadline:
        round_num += 1
        
        # Local search
        if time.time() < deadline - 5.0:
            local_best = _local_search(set(best_set), all_indices, triple_to_k222s, k222_list, 
                                       deadline - 2.0, max_iters=5000)
            if len(local_best) > best_size:
                best_size = len(local_best)
                best_set = set(local_best)
        
        # Simulated annealing every few rounds
        if round_num % 3 == 0 and time.time() < deadline - 10.0:
            sa_best = _simulated_annealing(set(best_set), all_indices, triple_to_k222s, k222_list,
                                           deadline - 5.0, T_start=0.5, T_end=0.01, cooling=0.9995)
            if len(sa_best) > best_size:
                best_size = len(sa_best)
                best_set = set(sa_best)
        
        # Perturbation and local search
        if round_num % 2 == 1 and time.time() < deadline - 15.0:
            perturbed = _perturb(best_set, all_indices, triple_to_k222s, k222_list, 
                                  n_remove=random.randint(2, 5), n_add=random.randint(2, 5))
            perturbed_best = _local_search(perturbed, all_indices, triple_to_k222s, k222_list,
                                           deadline - 5.0, max_iters=3000)
            if len(perturbed_best) > best_size:
                best_size = len(perturbed_best)
                best_set = set(perturbed_best)
        
        # Final polish with more iterations if time permits
        if time.time() < deadline - 10.0 and round_num % 5 == 0:
            polish_best = _local_search(set(best_set), all_indices, triple_to_k222s, k222_list,
                                        deadline - 2.0, max_iters=10000)
            if len(polish_best) > best_size:
                best_size = len(polish_best)
                best_set = set(polish_best)
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
