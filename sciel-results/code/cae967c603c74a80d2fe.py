"""Rewritten program with stronger algorithm for integer set pair-sum non-divisibility."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]

    # Precompute conflict matrix for evens: conflict[e1][e2] = True if (e1*e2)%(e1+e2)==0
    # Also precompute which odds each even conflicts with
    def conflicts(a, b):
        s = a + b
        p = a * b
        return p % s == 0

    # Build conflict sets
    even_conf_odds = {}   # e -> set of odds that conflict with e
    even_conf_evens = {}  # e -> set of evens that conflict with e

    for e in evens:
        oc = set()
        for o in odds:
            if conflicts(e, o):
                oc.add(o)
        even_conf_odds[e] = oc

        ec = set()
        for e2 in evens:
            if e2 != e and conflicts(e, e2):
                ec.add(e2)
        even_conf_evens[e] = ec

    # Odds that conflict with a given even
    # Actually, for the problem: we need (a*b) % (a+b) != 0 for all distinct pairs a,b in the set.
    # If we include all odds, we need to check odd-odd conflicts too.
    # Let's check: do any two odds conflict?
    odd_conflicts_exist = False
    for i in range(len(odds)):
        for j in range(i+1, len(odds)):
            a, b = odds[i], odds[j]
            if (a * b) % (a + b) == 0:
                odd_conflicts_exist = True
                break
        if odd_conflicts_exist:
            break

    # If there are odd-odd conflicts, we can't just take all odds.
    # Let's handle the general case: we need to find a large independent set in the conflict graph
    # on all of [1..256].

    # Precompute full conflict graph
    conflict_graph = {}  # x -> set of y where x!=y and (x*y)%(x+y)==0
    for x in all_nums:
        cs = set()
        for y in all_nums:
            if x == y:
                continue
            if (x * y) % (x + y) == 0:
                cs.add(y)
        conflict_graph[x] = cs

    # Strategy: Use a greedy + local search approach on the full conflict graph.
    # Key insight: the conflict graph structure may allow us to find a large independent set.
    
    # Precompute degree
    degrees = {x: len(conflict_graph[x]) for x in all_nums}
    
    # Greedy independent set with multiple orderings
    def greedy_independent_set(order):
        selected = set()
        for x in order:
            if not (conflict_graph[x] & selected):
                selected.add(x)
        return selected

    # Local search: try to add nodes, or swap
    def local_search(current_set, max_iter=500):
        if time.time() > DEADLINE:
            return current_set
        cur = set(current_set)
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < DEADLINE:
            improved = False
            iterations += 1
            
            # Try adding each unselected node
            for x in all_nums:
                if x in cur:
                    continue
                if not (conflict_graph[x] & cur):
                    cur.add(x)
                    improved = True
                    break
            
            # Try 1-for-1 swap: remove one, add another
            if not improved:
                for rem in list(cur):
                    trial = cur - {rem}
                    added_any = False
                    for x in all_nums:
                        if x in trial:
                            continue
                        if not (conflict_graph[x] & trial):
                            trial.add(x)
                            added_any = True
                            break
                    if added_any and len(trial) > len(cur):
                        cur = trial
                        improved = True
                        break
                # If no size increase, try 1-for-1 that keeps size but enables future adds? 
                # Skip for now.
            
            # Try 2-for-k swaps
            if not improved and time.time() < DEADLINE:
                cur_list = list(cur)
                n_cur = len(cur_list)
                for i in range(min(n_cur, 40)):
                    for j in range(i+1, min(n_cur, 40)):
                        rem1, rem2 = cur_list[i], cur_list[j]
                        trial = cur - {rem1, rem2}
                        added = 0
                        for x in all_nums:
                            if x in trial:
                                continue
                            if not (conflict_graph[x] & trial):
                                trial.add(x)
                                added += 1
                        if added > 2:
                            cur = trial
                            improved = True
                            break
                    if improved:
                        break
        
        return cur

    # Multiple greedy orderings
    best_set = set()
    best_size = 0

    # Ordering 1: by degree ascending
    order1 = sorted(all_nums, key=lambda x: (degrees[x], x))
    s1 = greedy_independent_set(order1)
    s1 = local_search(s1, max_iter=200)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1

    # Ordering 2: by value
    order2 = sorted(all_nums, key=lambda x: x)
    s2 = greedy_independent_set(order2)
    s2 = local_search(s2, max_iter=200)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # Ordering 3: by value descending
    order3 = sorted(all_nums, key=lambda x: -x)
    s3 = greedy_independent_set(order3)
    s3 = local_search(s3, max_iter=200)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3

    # Ordering 4: random
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        random_order = all_nums[:]
        rng.shuffle(random_order)
        sr = greedy_independent_set(random_order)
        sr = local_search(sr, max_iter=100)
        if len(sr) > best_size:
            best_size = len(sr)
            best_set = sr

    # More aggressive local search on the best
    if time.time() < DEADLINE:
        best_set = local_search(best_set, max_iter=1000)

    # Final verification
    result = sorted(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: just odds if they're valid
        if not odd_conflicts_exist:
            return odds
        else:
            # Return a single element
            return [1]

    return result
# EVOLVE-BLOCK-END
