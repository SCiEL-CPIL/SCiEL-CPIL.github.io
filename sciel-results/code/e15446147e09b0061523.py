"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s kill
    N = 256
    
    def reciprocal_sum(S):
        """Compute exact reciprocal sum as Fraction."""
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total
    
    def valid_addition(S, candidate):
        """
        Check if adding candidate to set S creates a 3-term AP.
        S is a set for O(1) lookups.
        """
        if candidate in S:
            return False
        
        # 1. Candidate is the middle term: a, candidate, c
        # a + c = 2 * candidate => c = 2 * candidate - a
        for a in S:
            c = 2 * candidate - a
            if c in S and c != a:
                return False
        
        # 2. Candidate is the first term: candidate, b, c
        # candidate + c = 2 * b => c = 2 * b - candidate
        for b in S:
            c = 2 * b - candidate
            if c in S and c != b:
                return False
        
        # 3. Candidate is the last term: a, b, candidate
        # a + candidate = 2 * b => a = 2 * b - candidate
        for b in S:
            a = 2 * b - candidate
            if a in S and a != b:
                return False
        
        return True
    
    def greedy_initial(order=None):
        """Greedy: add elements in given order (or 1..256) if valid."""
        if order is None:
            order = list(range(1, 257))
        selected = set()
        for x in order:
            if valid_addition(selected, x):
                selected.add(x)
        return selected
    
    def local_search(S, max_iter=100):
        """
        Local search with swaps and additions.
        S is a set.
        """
        S = set(S)
        best_sum = reciprocal_sum(S)
        best_S = set(S)
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            improved = False
            
            # Try single addition
            candidates_to_add = [x for x in range(1, 257) if x not in S]
            for cand in sorted(candidates_to_add, key=lambda x: -1/x):
                if valid_addition(S, cand):
                    new_S = S.union([cand])
                    new_sum = best_sum + Fraction(1, cand)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_S = new_S
                        S = new_S
                        improved = True
                        break
            
            # Try swaps: remove one, add one
            if not improved:
                S_list = list(S)
                for elem in S_list:
                    S_temp = S - {elem}
                    best_cand = None
                    for cand in range(1, elem):
                        if cand in S_temp:
                            continue
                        if valid_addition(S_temp, cand):
                            best_cand = cand
                            break
                    
                    if best_cand is not None:
                        new_S = S_temp.union([best_cand])
                        new_sum = best_sum - Fraction(1, elem) + Fraction(1, best_cand)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_S = new_S
                            S = new_S
                            improved = True
                            break
                
            if not improved:
                break
        
        return best_S
    
    # Phase 1: Greedy initial solution
    S1 = greedy_initial(order=list(range(1, 257)))
    S2 = greedy_initial(order=list(range(256, 0, -1)))
    S3 = greedy_initial(order=sorted(range(1, 257), key=lambda x: random.random()))
    
    best_S = max([S1, S2, S3], key=lambda s: reciprocal_sum(s))
    best_sum = reciprocal_sum(best_S)
    
    # Phase 2: Local search on the best initial
    best_S = local_search(best_S, max_iter=20)
    best_sum = reciprocal_sum(best_S)
    
    # Phase 3: Branch and Bound with bitset acceleration
    # Precompute conflict structure: for each pair (i,j), find k such that i,j,k form a 3-AP
    # conflict[i][j] = set of k values that would form a 3-AP with i and j
    
    # Actually, for B&B, we need: given a partial set S, which elements can we still add?
    # An element x can be added to S iff no two elements a,b in S form a 3-AP with x.
    # i.e., for all a in S: if (2*x - a) in S and (2*x - a) != a, then x conflicts.
    # Also: if a + x = 2*b for some b in S, conflict.
    # And: if x + b = 2*a for some a in S, conflict.
    
    # Let's use a different approach: maintain a "blocked" set.
    # When we add element x to S, it blocks certain pairs.
    # Actually, simpler: for B&B, we process elements in order of weight (1/x descending).
    # At each step, we decide to include or exclude the next element.
    
    # Precompute: for each x, the list of (a, b) pairs where {a,b,x} forms a 3-AP
    # If any of a,b is in S, then x cannot be added (unless the other is also not in S)
    # Wait, no: if a is in S and b is in S, then x conflicts.
    # So x is valid iff no pair (a,b) from its conflict list has both a and b in S.
    
    # Let's precompute conflicts[x] = list of (a, b) such that a,b,x form a 3-AP
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is the middle: a, x, c => a + c = 2x => c = 2x - a
        for a in range(1, x):
            c = 2 * x - a
            if c > x and c <= N:
                conflicts[x].append((a, c))
        # x is the first: x, b, c => x + c = 2b => c = 2b - x
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                conflicts[x].append((b, c))
        # x is the last: a, b, x => a + x = 2b => a = 2b - x
        for b in range(1, x):
            a = 2 * b - x
            if 0 < a < b:
                conflicts[x].append((a, b))
    
    # Remove duplicates
    for x in range(1, N + 1):
        conflicts[x] = list(set(conflicts[x]))
    
    # B&B: process elements in order of decreasing weight (increasing index)
    # We want to maximize sum of 1/x, so we consider x=1 first, then x=2, etc.
    # But actually, we should consider them in an order that helps pruning.
    # Let's try: consider elements in order 1, 2, 3, ..., 256.
    # At each step, we have a current set S and we decide to add or not add the next element.
    
    # For efficiency, we maintain:
    # - current_set: the set of elements chosen so far
    # - current_sum: the reciprocal sum so far
    # - upper_bound: an upper bound on what we can still achieve
    
    # Upper bound: current_sum + sum of 1/x for all remaining elements that could potentially be added
    # For a tighter bound, we could check which remaining elements are still compatible.
    
    def branch_and_bound(initial_set=None, deadline_check=None):
        """
        Branch and bound to find a good 3-AP-free set.
        Processes elements in order 1, 2, ..., 256.
        """
        if initial_set is None:
            initial_set = set()
        
        best_solution = set(initial_set)
        best_sum = reciprocal_sum(initial_set)
        
        # We'll do a recursive B&B with memoization of visited states?
        # That's too expensive. Instead, we'll do iterative deepening or just a single pass.
        
        # Actually, let's do a greedy-with-backtracking approach:
        # Try to build the set element by element, and if we get stuck, backtrack.
        
        # For simplicity and speed, let's do a limited-depth B&B.
        
        def bb_recursive(idx, current_set, current_sum, depth_limit=50):
            """
            Recursive B&B starting from element idx.
            current_set: set of chosen elements
            current_sum: Fraction sum so far
            depth_limit: how many more elements we can consider
            """
            nonlocal best_solution, best_sum
            
            if (time.time() - start_time) > DEADLINE:
                return
            
            if idx > N or depth_limit <= 0:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = set(current_set)
                return
            
            # Upper bound: current_sum + sum(1/x for x from idx to N)
            # This is a loose bound but fast to compute
            ub = current_sum
            for x in range(idx, N + 1):
                ub += Fraction(1, x)
            
            if ub <= best_sum:
                return  # Prune
            
            x = idx
            
            # Option 1: Include x
            if valid_addition(current_set, x):
                new_set = current_set | {x}
                new_sum = current_sum + Fraction(1, x)
                if new_sum + sum(Fraction(1, y) for y in range(idx + 1, N + 1)) > best_sum:
                    bb_recursive(idx + 1, new_set, new_sum, depth_limit - 1)
            
            # Option 2: Exclude x
            # Even if we exclude x, we might still achieve a better sum later
            # But the upper bound for excluding is the same ub (since we're not adding x)
            # Actually, the upper bound for excluding is current_sum + sum(1/x for x from idx+1 to N)
            ub_exclude = current_sum
            for x2 in range(idx + 1, N + 1):
                ub_exclude += Fraction(1, x2)
            if ub_exclude > best_sum:
                bb_recursive(idx + 1, current_set, current_sum, depth_limit)
        
        # Start B&B from the initial set
        # To keep it tractable, we only do B&B on a subset of elements
        # Let's try B&B on the first 30 elements (1-30) where weights are highest
        BB_LIMIT = 30
        bb_recursive(1, set(initial_set), reciprocal_sum(initial_set), BB_LIMIT)
        
        return best_solution, best_sum
    
    # Run B&B starting from the current best
    if (time.time() - start_time) < DEADLINE - 5:
        best_S, best_sum = branch_and_bound(initial_set=best_S)
    
    # Phase 4: Random restarts with local search
    restarts = 0
    while (time.time() - start_time) < DEADLINE - 2:
        restarts += 1
        order = list(range(1, 257))
        random.shuffle(order)
        S_new = greedy_initial(order=order)
        S_new = local_search(S_new, max_iter=10)
        new_sum = reciprocal_sum(S_new)
        if new_sum > best_sum:
            best_sum = new_sum
            best_S = S_new
    
    # Final polish
    best_S = local_search(best_S, max_iter=50)
    
    return list(best_S)
# EVOLVE-BLOCK-END
