import time
from math import gcd
from itertools import combinations, permutations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    def is_bad_triple(a, b, c):
        """Check if {a, b, c} form a bad triple."""
        return lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]

    def can_add(x, S):
        """Check if x can be added to set S without forming a bad triple."""
        # S is a list of elements
        for a, b in combinations(S, 2):
            if is_bad_triple(a, b, x):
                return False
        return True

    def greedy_solve(order):
        """Run greedy algorithm with a specific ordering of numbers."""
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        return S

    def local_search(S, R, time_limit):
        """
        Attempt to improve set S using local search.
        S: current valid set (list)
        R: list of rejected elements
        time_limit: absolute time to stop
        """
        S_set = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any element from R
            for x in R[:]:
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    R.remove(x)
                    improved = True
                    # Restart loop to check for new additions
                    break
            
            if time.time() > time_limit:
                break

            # If no direct additions, try swaps
            # For each x in R, find conflicts
            # If conflicts can be resolved by removing 1 element from S, do it
            for x in R[:]:
                if x in S_set:
                    continue
                
                # Find all pairs in S that conflict with x
                conflicts = []
                for a, b in combinations(S, 2):
                    if is_bad_triple(a, b, x):
                        conflicts.append((a, b))
                
                if not conflicts:
                    # Should have been caught by can_add, but just in case
                    S.append(x)
                    S_set.add(x)
                    R.remove(x)
                    improved = True
                    break
                
                # Check if removing 1 element resolves all conflicts
                # The element must be in every conflict pair
                # Intersection of all conflict pairs
                common = set(conflicts[0])
                for a, b in conflicts[1:]:
                    common.intersection_update({a, b})
                    if not common:
                        break
                
                if common:
                    # Try removing one of the common elements
                    for y in common:
                        # Check if S \ {y} is valid with x
                        # Since removing y doesn't create new triples, we just check if x is compatible with S \ {y}
                        # We know x conflicts with pairs containing y.
                        # Does x conflict with any pair NOT containing y?
                        # If all conflicts contain y, then S \ {y} has no conflicts with x.
                        # So S \ {y} U {x} is valid.
                        # Perform swap
                        S.remove(y)
                        S_set.remove(y)
                        S.append(x)
                        S_set.add(x)
                        R.remove(x)
                        # Add y back to R
                        R.append(y)
                        improved = True
                        break
                    if improved:
                        break
            
            if time.time() > time_limit:
                break

        return S

    best_set = []
    
    # Define multiple orderings
    orderings = [
        numbers,
        numbers[::-1],
        sorted(numbers, key=lambda x: x.bit_count(), reverse=True),
        sorted(numbers, key=lambda x: x.bit_count()),
    ]
    
    # Add some random orderings
    import random
    random.seed(42)
    for _ in range(20):
        shuffled = numbers[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline:
            break
        S = greedy_solve(order)
        if len(S) > len(best_set):
            best_set = S
    
    # Run local search on the best set
    if best_set:
        best_set_set = set(best_set)
        rejected = [x for x in numbers if x not in best_set_set]
        best_set = local_search(best_set, rejected, deadline)
    
    # If we have time, try more random restarts with local search
    while time.time() < deadline - 5:
        shuffled = numbers[:]
        random.shuffle(shuffled)
        S = greedy_solve(shuffled)
        if len(S) < len(best_set):
            continue
        
        S_set = set(S)
        rejected = [x for x in numbers if x not in S_set]
        S_improved = local_search(S, rejected, deadline - 2)
        
        if len(S_improved) > len(best_set):
            best_set = S_improved

    return best_set
