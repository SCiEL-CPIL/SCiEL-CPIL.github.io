import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations and triple-to-K mapping.
    Returns:
    - all_triples: list of tuples
    - triple_to_k222s: list of lists (for each triple index, list of K indices it belongs to)
    - k222_list: list of frozensets of triple indices
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_k222s = [[] for _ in range(len(all_triples))]
    k222_list = []
    
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # Use recursive generation
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
            # matching is list of 3 tuples: (a,a'), (b,b'), (c,c')
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

def _check_valid_set(selected_set, triple_to_k222s, k222_list):
    """Check if a set of triple indices is valid (no complete K(2,2,2))."""
    for k_idx in range(len(k222_list)):
        if k222_list[k_idx].issubset(selected_set):
            return False
    return True

def _greedy_with_weights(all_triples, triple_to_k222s, k222_list, order, weights=None):
    """
    Greedy construction: iterate through order, add triple if valid.
    If weights given, use them to break ties or prioritize.
    """
    if weights is None:
        weights = [1.0] * len(all_triples)
    
    present = set()
    for idx in order:
        # Check if adding idx is valid
        can_add = True
        for k_idx in triple_to_k222s[idx]:
            k_set = k222_list[k_idx]
            # Check if all other members of this K are in present
            for other_idx in k_set:
                if other_idx != idx and other_idx not in present:
                    break
            else:
                # All others present, can't add
                can_add = False
                break
        
        if can_add:
            present.add(idx)
    
    return present

def _local_search_swap(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=5.0):
    """
    Local search using 1-swap moves (remove one, add one).
    """
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    start_time = time.time()
    
    n_triples = len(all_triples)
    
    # Precompute for speed: for each triple, list of K indices it's in
    # Already have triple_to_k222s
    
    while time.time() - start_time < time_limit:
        current_list = list(current_set)
        if not current_list:
            break
        
        # Try a random swap
        u_idx = random.choice(current_list)
        
        # Pick random candidate to add
        t_idx = random.randint(0, n_triples - 1)
        attempts = 0
        while t_idx in current_set and attempts < 20:
            t_idx = random.randint(0, n_triples - 1)
            attempts += 1
        
        if t_idx in current_set:
            continue
        
        # Check if swap is valid
        valid_swap = True
        for k_idx in triple_to_k222s[t_idx]:
            k_set = k222_list[k_idx]
            # If u_idx is in this K, then after removing u_idx and adding t_idx,
            # the K is still missing u_idx, so it won't be complete.
            if u_idx in k_set:
                continue
            # Otherwise, check if all other members (except t_idx) are in current_set
            all_present = True
            for other in k_set:
                if other != t_idx and other not in current_set:
                    all_present = False
                    break
            if all_present:
                valid_swap = False
                break
        
        if valid_swap:
            current_set.discard(u_idx)
            current_set.add(t_idx)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
    
    return best_set

def _local_search_multi_remove(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=5.0):
    """
    Local search that sometimes removes 2 triples and adds 1-3 to escape local optima.
    """
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    start_time = time.time()
    n_triples = len(all_triples)
    
    while time.time() - start_time < time_limit:
        if len(current_set) < 2:
            break
        
        # Remove 1 or 2 random triples
        if random.random() < 0.3:
            # Remove 2
            if len(current_set) < 2:
                break
            current_list = list(current_set)
            u1 = random.choice(current_list)
            current_list.remove(u1)
            u2 = random.choice(current_list)
            removed = [u1, u2]
        else:
            # Remove 1
            u1 = random.choice(list(current_set))
            removed = [u1]
        
        trial_set = set(current_set)
        for r in removed:
            trial_set.discard(r)
        
        # Try adding 1-3 random triples
        num_add = len(removed) + random.randint(0, 1)
        added = []
        for _ in range(num_add):
            t_idx = random.randint(0, n_triples - 1)
            attempts = 0
            while (t_idx in trial_set or t_idx in added) and attempts < 20:
                t_idx = random.randint(0, n_triples - 1)
                attempts += 1
            if t_idx not in trial_set and t_idx not in added:
                # Check if adding t_idx to trial_set is valid
                can_add = True
                for k_idx in triple_to_k222s[t_idx]:
                    k_set = k222_list[k_idx]
                    for other in k_set:
                        if other != t_idx and other not in trial_set:
                            break
                    else:
                        can_add = False
                        break
                if can_add:
                    trial_set.add(t_idx)
                    added.append(t_idx)
        
        if len(trial_set) > len(current_set):
            current_set = trial_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Revert to current_set (trial_set was a copy, so just ignore)
            pass
    
    return best_set

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    n_triples = len(all_triples)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple random greedy restarts
    while time.time() - start_time < deadline - 30.0:
        order = list(range(n_triples))
        random.shuffle(order)
        
        g_set = _greedy_with_weights(all_triples, triple_to_k222s, k222_list, order)
        
        if len(g_set) > best_size:
            best_size = len(g_set)
            best_set = set(g_set)
        
        # Local search on this greedy result
        if time.time() - start_time < deadline - 15.0:
            ls_set = _local_search_swap(g_set, all_triples, triple_to_k222s, k222_list, time_limit=3.0)
            if len(ls_set) > best_size:
                best_size = len(ls_set)
                best_set = set(ls_set)
    
    # Phase 2: Intensive local search on best set found
    if time.time() - start_time < deadline - 5.0:
        final_set = _local_search_swap(best_set, all_triples, triple_to_k222s, k222_list, time_limit=8.0)
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)
    
    if time.time() - start_time < deadline - 5.0:
        final_set = _local_search_multi_remove(best_set, all_triples, triple_to_k222s, k222_list, time_limit=8.0)
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)
    
    # Final check and return
    if not _check_valid_set(best_set, triple_to_k222s, k222_list):
        # Should not happen, but if it does, fall back to empty
        best_set = set()
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
