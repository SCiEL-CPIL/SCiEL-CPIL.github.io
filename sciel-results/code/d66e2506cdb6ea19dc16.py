import time
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Strategy:
    1. Precompute bad triples and conflict pairs.
    2. Use a degree-based greedy algorithm to build the initial set.
    3. Apply local search and swaps to improve.
    """
    
    def is_valid_triple(a, b, c):
        """Check if three elements form a bad triple (all pairwise gcds equal)."""
        g_ab = gcd(a, b)
        g_ac = gcd(a, c)
        g_bc = gcd(b, c)
        return g_ab == g_ac == g_bc
    
    # Precompute all bad triples
    bad_triples = []
    all_nums = list(range(1, 129))
    for i in range(len(all_nums)):
        for j in range(i+1, len(all_nums)):
            for k in range(j+1, len(all_nums)):
                if is_valid_triple(all_nums[i], all_nums[j], all_nums[k]):
                    bad_triples.append((all_nums[i], all_nums[j], all_nums[k]))
    
    # Compute degree for each number (count of bad triples it is in)
    degrees = {x: 0 for x in all_nums}
    for a, b, c in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    # Precompute conflict pairs for fast checking
    # x_conflict_pairs[x] is a list of (y, z) such that (x, y, z) is a bad triple
    x_conflict_pairs = {x: [] for x in all_nums}
    for a, b, c in bad_triples:
        x_conflict_pairs[a].append((b, c))
        x_conflict_pairs[b].append((a, c))
        x_conflict_pairs[c].append((a, b))
        
    def count_conflicts(x, selected):
        """Count how many bad triplets would be formed if x were added to selected."""
        conflicts = 0
        for y, z in x_conflict_pairs[x]:
            if y in selected and z in selected:
                conflicts += 1
        return conflicts

    def greedy_build_degree(order):
        """
        Build a valid set by adding elements in the given order.
        """
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        return selected

    def local_search(selected, max_iters=50):
        """Try to improve the set by removing/adding elements."""
        selected = set(selected)
        for _ in range(max_iters):
            improved = False
            # Try adding each excluded element, sorted by degree (lowest first)
            excluded = [x for x in range(1, 129) if x not in selected]
            excluded.sort(key=lambda x: degrees[x])
            
            for x in excluded:
                if count_conflicts(x, selected) == 0:
                    selected.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try to remove an element to allow adding others
                # Heuristic: Remove element with highest degree in the current set?
                best_remove = None
                best_new_set = None
                best_size = len(selected)
                
                # Limit search to elements with high degree to save time
                selected_list = sorted(list(selected), key=lambda x: degrees[x], reverse=True)
                
                for x in selected_list:
                    temp = selected - {x}
                    # Try to add elements to temp
                    new_selected = set(temp)
                    # Try adding all excluded elements (sorted by degree)
                    for y in excluded:
                        if count_conflicts(y, new_selected) == 0:
                            new_selected.add(y)
                            
                    if len(new_selected) > best_size:
                        best_size = len(new_selected)
                        best_remove = x
                        best_new_set = new_selected
                        break
                
                if best_remove is not None:
                    selected = best_new_set
                    improved = True
            
            if not improved:
                break
        return selected

    deadline = time.time() + 85.0
    best_set = set()
    best_size = 0
    
    import random
    random.seed(42)
    
    # Strategy 1: Degree-based greedy (ascending degree)
    sorted_by_degree = sorted(all_nums, key=lambda x: degrees[x])
    s1 = greedy_build_degree(sorted_by_degree)
    s1 = local_search(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
        
    # Strategy 2: Randomized Degree-based greedy
    for trial in range(100):
        if time.time() > deadline:
            break
        shuffled = all_nums[:]
        random.shuffle(shuffled)
        # Stable sort by degree
        shuffled.sort(key=lambda x: degrees[x])
        s2 = greedy_build_degree(shuffled)
        s2 = local_search(s2)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2
            
    # Strategy 3: Local search with swaps (2-1 exchange)
    for _ in range(20):
        if time.time() > deadline:
            break
        current = set(best_set)
        excluded = [x for x in range(1, 129) if x not in current]
        excluded.sort(key=lambda x: degrees[x])
        
        improved = False
        for x in list(current):
            temp = current - {x}
            added = []
            # Try to add excluded elements to temp
            for y in excluded:
                if count_conflicts(y, temp | set(added)) == 0:
                    added.append(y)
                    if len(added) >= 2: # If we add 2, we gain 1 net.
                        break
            if len(added) >= 2:
                new_set = temp | set(added)
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
                    break
            
        if improved:
            best_set = current
            best_size = len(current)
        else:
            # Try a random restart of local search
            if best_set:
                perturbed = set(best_set)
                if perturbed:
                    perturbed.remove(random.choice(list(perturbed)))
                perturbed = local_search(perturbed)
                if len(perturbed) > best_size:
                    best_size = len(perturbed)
                    best_set = perturbed

    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
