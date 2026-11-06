import math
import random
import time
from collections import defaultdict

def construct():
    N = 128
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Precompute LCM table
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)
    
    # Precompute bad_pairs
    # bad_pairs[x] = set of (a,b) with a<b, a!=x, b!=x such that lcm(a,b)==lcm(a,x)==lcm(b,x)
    bad_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if lcm[a][b] == lcm[a][x] == lcm[b][x]:
                    bad_pairs[x].add((a, b))
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without forming a bad triple."""
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy_construct(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        for x in order:
            if x not in selected_set and can_add(selected_set, x):
                selected_set.add(x)
        return selected_set

    # 1. Initial Greedy
    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    # Try a few specific orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1], # Odds
        [x for x in range(1, N + 1) if x % 2 == 0], # Evens
    ]
    for _ in range(50):
        orders.append(rng.sample(range(1, N + 1), N))
        
    for order in orders:
        if time.time() > deadline: break
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
    current_set = best_set
    current_size = len(current_set)
    all_nums = set(range(1, N + 1))
    
    # Tabu Search
    tabu_tenure = 10
    tabu_list = {} # element -> time until not tabu
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any missing element (Greedy Add)
        missing = list(all_nums - current_set)
        rng.shuffle(missing)
        for v in missing:
            if can_add(current_set, v):
                current_set.add(v)
                current_size += 1
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
            
        # If no add possible, try Tabu Swaps
        # Generate candidate moves: remove u, add v
        # We want to maximize size, so swaps that keep size same are neutral, 
        # but might enable future adds.
        # Swaps that increase size are best.
        
        candidates = []
        selected_list = list(current_set)
        missing_list = list(all_nums - current_set)
        
        if not missing_list:
            break
            
        # Limit candidates to speed up
        # Sample a subset of selected and missing if lists are large
        # N=128, so full list is small enough.
        
        best_move_score = -1
        best_u = None
        best_v = None
        
        # To speed up, we don't evaluate ALL N*N swaps.
        # We can sample.
        sample_size = min(len(selected_list), 20)
        sample_missing_size = min(len(missing_list), 20)
        
        sampled_selected = rng.sample(selected_list, sample_size)
        sampled_missing = rng.sample(missing_list, sample_missing_size)
        
        for u in sampled_selected:
            if u in tabu_list and tabu_list[u] > 0:
                continue
            for v in sampled_missing:
                if v in tabu_list and tabu_list[v] > 0:
                    continue
                # Try swap u -> v
                # Check if v can be added to current_set - {u}
                temp_set = current_set - {u}
                if can_add(temp_set, v):
                    # Size remains same
                    # We need a heuristic to pick which swap is "better"
                    # Maybe count how many new elements become addable?
                    # That's expensive.
                    # Let's just accept random valid swaps to explore.
                    # Or, check if v has fewer conflicts than u?
                    # Let's just pick the first valid one or random.
                    # To be smarter, let's calculate "potential"
                    # Potential = number of elements w in missing \ {v} such that can_add(temp_set U {v}, w)
                    # This is O(N) per swap, which is 128 * 400 = 50k ops. Might be slow.
                    
                    # Simpler: Just accept the first valid swap found.
                    # This is random walk.
                    # Let's use a score: -1 (neutral)
                    # We will accept it.
                    best_u = u
                    best_v = v
                    best_move_score = 0
                    break
            if best_u is not None:
                break
                
        if best_u is not None and best_v is not None:
            # Perform swap
            current_set.remove(best_u)
            current_set.add(best_v)
            # Update tabu lists
            tabu_list[best_u] = tabu_tenure
            tabu_list[best_v] = tabu_tenure
            no_improve_count += 1
        else:
            # No valid swap found in sample
            # Try a "kick": remove k random elements and greedily fill
            if no_improve_count > max_no_improve:
                k = 3
                to_remove = rng.sample(list(current_set), min(k, len(current_set)))
                for u in to_remove:
                    current_set.remove(u)
                # Greedy fill
                missing = list(all_nums - current_set)
                rng.shuffle(missing)
                for v in missing:
                    if can_add(current_set, v):
                        current_set.add(v)
                current_size = len(current_set)
                no_improve_count = 0
                # Clear tabu
                tabu_list = {}
                
        # Decay tabu lists
        for u in list(tabu_list.keys()):
            tabu_list[u] -= 1
            if tabu_list[u] <= 0:
                del tabu_list[u]
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            no_improve_count = 0

    if not best_set:
        best_set = {1}
        
    return list(best_set)
