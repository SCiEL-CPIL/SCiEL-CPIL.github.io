"""Max-SAT style local search for large integer sets avoiding equal-gcd triples."""
import time
import random
from math import gcd
from itertools import combinations

N = 128

def _build_gcd_table():
    """Precompute gcd for all pairs in [1, 128]."""
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    return g

def _is_bad_triple(i, j, k, g):
    """Check if triple (i,j,k) has all three pairwise gcds equal."""
    if i == j or j == k or i == k:
        return False
    gij = g[i][j]
    gik = g[i][k]
    gjk = g[j][k]
    return gij == gik == gjk

def _count_bad_in_set(S, g):
    """Count bad triples in set S."""
    count = 0
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            for k in range(j + 1, len(S_list)):
                if _is_bad_triple(S_list[i], S_list[j], S_list[k], g):
                    count += 1
    return count

def _greedy_initial(g):
    """Greedy construction: add numbers in order if they don't create a bad triple."""
    S = set()
    for x in range(1, N + 1):
        ok = True
        for a, b in combinations(S, 2):
            if _is_bad_triple(a, b, x, g):
                ok = False
                break
        if ok:
            S.add(x)
    return S

def _local_search(g, S, time_limit, seed=0):
    """
    Local search: try to improve set S by swapping elements.
    A move is: remove one element from S, add one from complement.
    Accept if it reduces the number of bad triples (or keeps same and we want larger set).
    Actually, we want a VALID set (0 bad triples) and maximize size.
    
    Strategy: 
    1. Start with a valid set from greedy.
    2. Try to add elements from complement that don't create bad triples.
    3. If stuck, try swapping: remove an element that's 'blocking' many additions, 
       add a different element.
    
    Better: Use a penalty-based approach. Minimize (number_of_bad_triples * BIG_PENALTY - |S|).
    Use simulated annealing or hill climbing on this objective.
    """
    rng = random.Random(seed)
    S = set(S)
    best_S = set(S)
    best_score = _score(S, g)
    
    # Precompute for each x in [1..N], the list of pairs (a,b) in current S 
    # such that (a,b,x) is a bad triple. This is expensive to maintain dynamically.
    # Instead, for each candidate move, compute the change in bad triples.
    
    # For efficiency, let's use a different approach:
    # Objective: maximize |S| subject to 0 bad triples.
    # We can do: 
    #   Phase 1: Get a valid set via greedy.
    #   Phase 2: Try to add any element from complement that keeps validity.
    #   Phase 3: If no single addition works, try swaps: remove x from S, add y from complement,
    #            if the result is valid and |S| stays same (swap) or we can then add more.
    
    # Let's implement a more systematic local search:
    # State: S (set of selected numbers)
    # Move: pick x in S, y not in S. New S' = S - {x} + {y}.
    # Accept if S' is valid (0 bad triples) and |S'| >= |S| (always true for swap).
    # But we want to maximize size, so we prefer moves that allow future additions.
    
    # Alternative: Penalty method.
    # Score(S) = -|S| if valid, else -|S| - 1000 * (# bad triples)
    # Minimize Score.
    
    def score(S):
        if not S:
            return 0
        # Count bad triples
        bad = 0
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if _is_bad_triple(S_list[i], S_list[j], S_list[k], g):
                        bad += 1
        if bad == 0:
            return -len(S)
        else:
            return -len(S) - 1000 * bad
    
    current_S = set(S)
    current_score = score(current_S)
    best_S = set(current_S)
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        if len(current_S) == 0:
            # Add a random element
            y = rng.randint(1, N)
            new_S = current_S | {y}
        else:
            # Pick x to remove, y to add
            x = rng.choice(list(current_S))
            complement = set(range(1, N+1)) - current_S
            if not complement:
                break
            y = rng.choice(list(complement))
            new_S = current_S - {x} | {y}
        
        new_score = score(new_S)
        
        # Accept if better, or with some probability if worse (simulated annealing)
        if new_score < current_score:
            current_S = new_S
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_S = set(current_S)
        else:
            no_improve_count += 1
            # Simulated annealing acceptance
            if no_improve_count > max_no_improve:
                # Restart with a perturbation
                # Remove a random element and add a random one
                if current_S:
                    x = rng.choice(list(current_S))
                    complement = set(range(1, N+1)) - current_S
                    if complement:
                        y = rng.choice(list(complement))
                        current_S = current_S - {x} | {y}
                        current_score = score(current_S)
                no_improve_count = 0
            else:
                # Try to accept worse moves with decreasing probability
                delta = new_score - current_score
                if delta < 0:
                    pass
                else:
                    # Probability of accepting worse move
                    # Use a simple heuristic: accept with probability 1/(1+delta/100)
                    p_accept = 1.0 / (1.0 + max(0, delta) / 100.0)
                    if rng.random() < p_accept:
                        current_S = new_S
                        current_score = new_score
                    else:
                        # Try to fix validity: if current_S has bad triples, try to remove one element involved
                        if current_score > -len(current_S):  # has bad triples
                            # Find a bad triple and remove one of its elements
                            S_list = list(current_S)
                            found = False
                            for i in range(len(S_list)):
                                for j in range(i+1, len(S_list)):
                                    for k in range(j+1, len(S_list)):
                                        if _is_bad_triple(S_list[i], S_list[j], S_list[k], g):
                                            # Remove one of them
                                            remove_candidate = rng.choice([S_list[i], S_list[j], S_list[k]])
                                            current_S = current_S - {remove_candidate}
                                            current_score = score(current_S)
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if not found:
                                # No bad triples, but we didn't improve size? Shouldn't happen
                                pass
    
    return best_S

def _try_additions(S, g):
    """Try to add any element from complement that keeps the set valid."""
    S = set(S)
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            # Check if adding x creates any bad triple
            ok = True
            for a, b in combinations(S, 2):
                if _is_bad_triple(a, b, x, g):
                    ok = False
                    break
            if ok:
                S.add(x)
                changed = True
                break  # Restart loop since S changed
    return S

def construct():
    """
    Return a nonempty subset of [1, 128] with no three distinct elements
    whose three pairwise gcds are equal. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s safety margin
    
    g = _build_gcd_table()
    
    # Phase 1: Greedy initial solution
    S = _greedy_initial(g)
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Try to add more elements
    S = _try_additions(S, g)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Phase 3: Local search with time budget
    time_for_local_search = min(70, max(5, (deadline - time.time()) - 5))
    
    # Multiple restarts with different seeds
    seed = 0
    while time.time() < deadline - 5:
        # Run local search
        S_ls = _local_search(g, best_S, time_for_local_search, seed)
        
        # Try to add more elements after local search
        S_ls = _try_additions(S_ls, g)
        
        if len(S_ls) > best_size:
            best_S = set(S_ls)
            best_size = len(S_ls)
        
        seed += 1
        # Reduce time for subsequent runs
        time_for_local_search = max(2, time_for_local_search * 0.8)
    
    # Final: try additions one more time
    final_S = _try_additions(best_S, g)
    if len(final_S) > best_size:
        best_S = final_S
    
    # Ensure nonempty
    if not best_S:
        best_S = {1}
    
    return best_S

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
