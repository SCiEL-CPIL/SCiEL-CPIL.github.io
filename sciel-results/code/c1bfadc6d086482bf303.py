"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200.
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 45  # Safety margin before 60s limit
    
    def is_valid(selected_set):
        """Check if no subset of selected_set sums to 200 using bitset DP."""
        reachable = [False] * 201
        reachable[0] = True
        for x in selected_set:
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        return not reachable[200]
    
    # Strategy 1: Threshold set {66..100} - guaranteed valid, 35 elements
    threshold_set = list(range(66, 101))
    
    # Strategy 2: Greedy from largest to smallest
    greedy_selected = []
    reachable_greedy = {0}
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        target = 200 - x
        if target in reachable_greedy:
            continue
        new_sums = {s + x for s in reachable_greedy if s + x < 200}
        greedy_selected.append(x)
        reachable_greedy |= new_sums
    
    # Strategy 3: Start with threshold set, try to add smaller elements
    extended = list(threshold_set)
    reach_ext = [False] * 201
    reach_ext[0] = True
    for x in extended:
        for s in range(200 - x, -1, -1):
            if reach_ext[s]:
                reach_ext[s + x] = True
    
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        if not reach_ext[200 - x]:
            extended.append(x)
            for s in range(200 - x, -1, -1):
                if reach_ext[s]:
                    reach_ext[s + x] = True
    
    # Strategy 4: Greedy from smallest to largest
    greedy_small = []
    reachable_small = {0}
    for x in range(1, 101):
        if time.time() > deadline:
            break
        target = 200 - x
        if target in reachable_small:
            continue
        new_sums = {s + x for s in reachable_small if s + x < 200}
        greedy_small.append(x)
        reachable_small |= new_sums
    
    # Strategy 5: Backtracking with forward checking
    # We want to maximize cardinality. Process elements in descending order
    # (larger first) to prioritize including them, which tends to create
    # conflicts that prune the search space more effectively.
    
    def backtracking_search(elements, current_set, reachable, depth, best_size, best_set):
        """
        Backtracking search with forward checking.
        elements: list of remaining elements to consider (sorted descending)
        current_set: list of elements included so far
        reachable: set of reachable sums with current_set
        depth: current depth in search
        best_size: current best cardinality found
        best_set: current best set found
        
        Returns (best_size, best_set)
        """
        if time.time() > deadline:
            return best_size, best_set
        
        # Pruning: even if we include all remaining elements, can we beat best?
        if len(current_set) + len(elements) <= best_size:
            return best_size, best_set
        
        if not elements:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return best_size, best_set
        
        x = elements[0]
        rest = elements[1:]
        
        # Option 1: Include x if it doesn't create sum 200
        if (200 - x) not in reachable:
            # Forward check: adding x is safe
            new_reachable = set(reachable)
            for s in list(reachable):
                ns = s + x
                if ns < 200:
                    new_reachable.add(ns)
            # Check if any sum reaches 200
            if 200 not in new_reachable:
                current_set.append(x)
                best_size, best_set = backtracking_search(
                    rest, current_set, new_reachable, depth + 1, best_size, best_set
                )
                current_set.pop()
        
        # Option 2: Exclude x
        best_size, best_set = backtracking_search(
            rest, current_set, reachable, depth + 1, best_size, best_set
        )
        
        return best_size, best_set
    
    # Run backtracking with descending order
    bt_elements = list(range(100, 0, -1))
    bt_size, bt_set = backtracking_search(
        bt_elements, [], {0}, 0, 0, []
    )
    
    # Strategy 6: Local search / hill climbing to improve
    # Try to add more elements to the best set found
    def try_improve(base_set, base_reachable):
        """Try to add more elements to base_set without creating sum 200."""
        improved = list(base_set)
        reach = set(base_reachable)
        
        # Try adding elements not in base_set, from largest to smallest
        for x in range(100, 0, -1):
            if time.time() > deadline:
                break
            if x in improved:
                continue
            if (200 - x) not in reach:
                # Safe to add
                new_reach = set(reach)
                for s in list(reach):
                    ns = s + x
                    if ns < 200:
                        new_reach.add(ns)
                if 200 not in new_reach:
                    improved.append(x)
                    reach = new_reach
        
        return improved
    
    # Improve the backtracking result
    if bt_set:
        # Compute reachable for bt_set
        bt_reach = {0}
        for x in bt_set:
            new_sums = {s + x for s in bt_reach if s + x < 200}
            bt_reach |= new_sums
        improved_bt = try_improve(bt_set, bt_reach)
    else:
        improved_bt = []
    
    # Improve the extended set
    ext_reach = {0}
    for x in extended:
        new_sums = {s + x for s in ext_reach if s + x < 200}
        ext_reach |= new_sums
    improved_ext = try_improve(extended, ext_reach)
    
    # Improve the greedy_high set
    gh_reach = {0}
    for x in greedy_selected:
        new_sums = {s + x for s in gh_reach if s + x < 200}
        gh_reach |= new_sums
    improved_gh = try_improve(greedy_selected, gh_reach)
    
    # Pick the best valid set
    candidates = [
        ("threshold", threshold_set),
        ("greedy_high", greedy_selected),
        ("extended", extended),
        ("greedy_low", greedy_small),
        ("backtracking", bt_set),
        ("improved_bt", improved_bt),
        ("improved_ext", improved_ext),
        ("improved_gh", improved_gh),
    ]
    
    best = None
    best_size = 0
    for name, cand in candidates:
        if not cand:
            continue
        if is_valid(cand):
            if len(cand) > best_size:
                best_size = len(cand)
                best = cand
    
    if best is None:
        return [100]
    
    return best
# EVOLVE-BLOCK-END
