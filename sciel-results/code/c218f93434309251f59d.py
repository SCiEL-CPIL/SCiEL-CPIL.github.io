import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    N = 256
    
    # Precompute prime factors for 1..N using sieve
    F = [None] * (N + 1)
    F[1] = frozenset()
    
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for i in range(2, N + 1):
        x = i
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        F[i] = frozenset(factors)
    
    # Precompute coprime compatibility: coprime[i][j] = True if gcd(i,j)==1
    # Use prime factor sets for efficiency
    def are_coprime(a, b):
        return F[a].isdisjoint(F[b])
    
    # Upper bound helper: sum of all candidates that are individually coprime to current set
    def compute_upper_bound(candidates, current_factors):
        """Sum of candidates whose prime factors are disjoint from current_factors."""
        total = 0
        for c in candidates:
            if F[c].isdisjoint(current_factors):
                total += c
        return total
    
    def branch_and_bound(candidates, current_sum, current_factors, current_set, best_sum, best_set):
        """
        Exact branch-and-bound for maximum weight independent set.
        candidates: list of remaining candidate numbers (sorted descending)
        current_sum: sum of selected numbers so far
        current_factors: set of prime factors used
        current_set: list of selected numbers
        Returns: (best_sum, best_set)
        """
        if not candidates:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = current_set[:]
            return best_sum, best_set
        
        # Upper bound: current_sum + sum of all candidates coprime to current_factors
        ub = current_sum
        for c in candidates:
            if F[c].isdisjoint(current_factors):
                ub += c
        
        if ub <= best_sum[0]:
            return best_sum, best_set
        
        # Try including the first candidate
        c0 = candidates[0]
        f0 = F[c0]
        
        if f0.isdisjoint(current_factors):
            # Include c0
            new_factors = current_factors | f0
            new_set = current_set + [c0]
            # Filter remaining candidates to those coprime to new_factors
            remaining = [c for c in candidates[1:] if F[c].isdisjoint(new_factors)]
            branch_and_bound(remaining, current_sum + c0, new_factors, new_set, best_sum, best_set)
        
        # Try excluding c0
        branch_and_bound(candidates[1:], current_sum, current_factors, current_set, best_sum, best_set)
        
        return best_sum, best_set
    
    def greedy_solve(candidates, current_factors=None):
        """Greedy: pick largest coprime numbers."""
        if current_factors is None:
            current_factors = set()
        selected = []
        for c in sorted(candidates, reverse=True):
            if F[c].isdisjoint(current_factors):
                selected.append(c)
                current_factors = current_factors | F[c]
        return selected
    
    def local_improve(selected):
        """Try to improve a solution by swapping elements."""
        selected = list(selected)
        sel_set = set(selected)
        sel_factors = set()
        for x in selected:
            sel_factors.update(F[x])
        
        improved = True
        while improved:
            improved = False
            # Try adding any number that fits
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if F[x].isdisjoint(sel_factors):
                    selected.append(x)
                    sel_set.add(x)
                    sel_factors.update(F[x])
                    improved = True
                    break
            
            # Try 1-swap: remove one, add best replacement
            if not improved:
                for i, a in enumerate(selected):
                    rest = selected[:i] + selected[i+1:]
                    rest_factors = set()
                    for x in rest:
                        rest_factors.update(F[x])
                    
                    # Find best single replacement
                    best_rep = None
                    for x in range(1, N + 1):
                        if x in set(rest):
                            continue
                        if F[x].isdisjoint(rest_factors):
                            if best_rep is None or x > best_rep:
                                best_rep = x
                    
                    if best_rep is not None and best_rep > a:
                        selected[i] = best_rep
                        sel_set = set(selected)
                        sel_factors = set()
                        for x in selected:
                            sel_factors.update(F[x])
                        improved = True
                        break
        
        return selected
    
    # Main search strategy
    best_sum = 0
    best_result = [1]
    
    # Phase 1: Multiple greedy initializations with different orderings
    orderings = []
    
    # Descending
    orderings.append(list(range(N, 0, -1)))
    
    # Randomized orderings
    rng = random.Random(42)
    for _ in range(10):
        orderings.append(list(range(1, N + 1)))
        rng.shuffle(orderings[-1])
    
    # Value/weight ratio ordering (value / number_of_prime_factors)
    ratio_order = list(range(N, 0, -1))
    ratio_order.sort(key=lambda x: x / max(1, len(F[x])), reverse=True)
    orderings.append(ratio_order)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        # Greedy with this ordering
        selected = []
        used_factors = set()
        for c in ordering:
            if F[c].isdisjoint(used_factors):
                selected.append(c)
                used_factors = used_factors | F[c]
        
        # Local improvement
        selected = local_improve(selected)
        s = sum(selected)
        
        if s > best_sum:
            best_sum = s
            best_result = selected[:]
    
    # Phase 2: Exact branch-and-bound on the remaining problem
    # Take the best solution and try to improve via exact search on small subproblems
    if time.time() < deadline - 15:
        # Try exact B&B on a subset of candidates
        # Use the best solution's factors to define the problem
        best_factors = set()
        for x in best_result:
            best_factors.update(F[x])
        
        # Find candidates that could potentially improve
        # A candidate c improves if replacing some subset gives higher sum
        # Simplified: try adding one candidate at a time and see if we can remove others
        for c in range(N, 0, -1):
            if time.time() > deadline - 5:
                break
            if c in set(best_result):
                continue
            if not F[c].isdisjoint(best_factors):
                continue
            
            # Try to add c: which existing elements conflict?
            conflicts = [x for x in best_result if not F[x].isdisjoint(F[c])]
            if len(conflicts) > 2:
                continue
            
            # Try removing conflicts and adding c
            new_set = [x for x in best_result if x not in conflicts] + [c]
            new_sum = sum(new_set)
            
            if new_sum > best_sum:
                # Verify and try to fill gaps
                new_set = local_improve(new_set)
                new_sum = sum(new_set)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_result = new_set[:]
    
    # Phase 3: Final exact refinement if time permits
    if time.time() < deadline - 5:
        # Try a few random restarts with exact B&B on reduced problem
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random subset of candidates
            subset = list(range(1, N + 1))
            random.shuffle(subset)
            subset = subset[:40]  # Limit to 40 candidates for tractability
            
            best_sub = [0]
            best_sub_set = [[]]
            branch_and_bound(
                sorted(subset, reverse=True),
                0,
                set(),
                [],
                best_sub,
                best_sub_set
            )
            
            if best_sub[0] > best_sum:
                best_sum = best_sub[0]
                best_result = best_sub_set[0][:]
    
    # Ensure 1 is included
    if 1 not in best_result:
        best_result.append(1)
    
    # Final verification
    n_res = len(best_result)
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if math.gcd(best_result[i], best_result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not (valid and len(set(best_result)) == len(best_result) and all(1 <= x <= N for x in best_result)):
        # Fallback: 1 plus all primes
        primes = [i for i in range(2, N + 1) if F[i] and len(F[i]) == 1 and min(F[i]) == i]
        best_result = [1] + primes
    
    # Type-check
    final = [int(x) for x in best_result if isinstance(x, int) and not isinstance(x, bool)]
    if not final:
        final = [1]
    
    return final
