"""Combined program: precomputed intersections + grouping conflict check + local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)  # C(16,4) = 1820
    
    # Precompute pairwise intersections
    pair_int = [[0] * num_subsets for _ in range(num_subsets)]
    for i in range(num_subsets):
        si = all_subsets[i]
        for j in range(i + 1, num_subsets):
            v = si & all_subsets[j]
            pair_int[i][j] = v
            pair_int[j][i] = v
    
    def can_add(new_idx, family_list, family_set):
        """Check if new_idx can be added to family without forming a sunflower."""
        s = all_subsets[new_idx]
        # Group family members by their intersection with s
        groups = defaultdict(list)
        for fi in family_list:
            inter = all_subsets[fi] & s
            groups[inter].append(fi)
        
        for inter, members in groups.items():
            if len(members) < 2:
                continue
            L = len(members)
            for i in range(L):
                gi = members[i]
                for j in range(i + 1, L):
                    gj = members[j]
                    if pair_int[gi][gj] == inter:
                        return False
        return True
    
    best_family = []
    best_size = 0
    
    def greedy_construct(order, time_limit):
        """Run greedy construction with given order."""
        family = []
        family_set = set()
        for idx in order:
            if time.time() > time_limit:
                break
            if idx in family_set:
                continue
            if can_add(idx, family, family_set):
                family.append(idx)
                family_set.add(idx)
        return family, family_set
    
    def local_search(family, family_set, time_limit):
        """Improve family via add and swap moves."""
        family = list(family)
        family_set = set(family)
        improved = True
        max_iters = 50
        
        while improved and time.time() < time_limit and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try adding any set that doesn't conflict
            for idx in range(num_subsets):
                if time.time() > time_limit:
                    break
                if idx in family_set:
                    continue
                if can_add(idx, family, family_set):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one
            # Shuffle family to vary which one we remove
            shuffled_family = family[:]
            random.shuffle(shuffled_family)
            for remove_idx in shuffled_family:
                if time.time() > time_limit:
                    break
                
                temp_family = [x for x in family if x != remove_idx]
                temp_set = set(temp_family)
                
                # Find a candidate to add
                for add_idx in range(num_subsets):
                    if add_idx in temp_set:
                        continue
                    if can_add(add_idx, temp_family, temp_set):
                        family = temp_family
                        family.append(add_idx)
                        family_set = set(family)
                        improved = True
                        break
                
                if improved:
                    break
        
        return family, family_set
    
    # Strategy: Multiple greedy passes with different orderings
    # Then local search on the best result
    
    # First: lexicographic order
    order_lex = list(range(num_subsets))
    family, fset = greedy_construct(order_lex, deadline - 20)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]
    
    # Reverse lexicographic
    order_rev = list(range(num_subsets - 1, -1, -1))
    family, fset = greedy_construct(order_rev, deadline - 20)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]
    
    # Random restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        random.seed(42 + restart)
        order_rand = list(range(num_subsets))
        random.shuffle(order_rand)
        
        family, fset = greedy_construct(order_rand, deadline - 25)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search on the best family found
    if time.time() < deadline - 10 and best_size > 0:
        best_family_set = set(best_family)
        best_family, best_family_set = local_search(best_family, best_family_set, deadline - 5)
        best_size = len(best_family)
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
