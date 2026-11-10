"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # Find all bad triples: {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Build conflict graph: edge between x,y if there exists z such that {x,y,z} is a bad triple
    # Actually, we need: no three distinct elements form a bad triple.
    # So the "forbidden" configuration is a triple {a,b,c} with equal pairwise LCMs.
    # We can model this as: create a hypergraph where each bad triple is a hyperedge.
    # For Maximum Independent Set, we need a graph. 
    # Alternative: two vertices are "conflicting" if they appear together in some bad triple
    # AND that triple's third element is also present. But that's not pairwise.
    # 
    # Better approach: The constraint is that NO three elements form a bad triple.
    # This is a 3-uniform hypergraph independent set problem.
    # For branch-and-bound, we can use the following:
    # A set S is valid iff for every pair (a,b) in S, there is no c in S such that 
    # lcm(a,b)==lcm(a,c)==lcm(b,c).
    # 
    # Let's precompute: for each pair (a,b), the set of "witnesses" W(a,b) = 
    # {c : lcm(a,b)==lcm(a,c)==lcm(b,c)}
    # Then S is valid iff for all distinct a,b,c in S, c notin W(a,b).
    # 
    # For the conflict graph approach: we can't directly reduce to a graph MIS easily.
    # Instead, let's use a different exact approach: branch and bound with smart pruning.
    # 
    # Actually, let me reconsider. The problem is to maximize |S| such that no 
    # {a,b,c} subset of S has lcm(a,b)=lcm(a,c)=lcm(b,c).
    # 
    # Let's precompute all bad triples first.
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_triples.add((a, b, c))
    
    # Now, build a conflict structure: for each pair (a,b), which c's would make a bad triple
    pair_witness = defaultdict(set)
    for (a, b, c) in bad_triples:
        pair_witness[(a, b)].add(c)
        pair_witness[(b, a)].add(c)
        pair_witness[(a, c)].add(b)
        pair_witness[(c, a)].add(b)
        pair_witness[(b, c)].add(a)
        pair_witness[(c, b)].add(a)
    
    # For branch and bound, we need to check if adding x to current set S is valid.
    # x can be added iff for all pairs (a,b) in S, x notin pair_witness[(a,b)]
    # i.e., x is not a witness for any pair in S.
    
    # Precompute: for each x, which pairs does x witness?
    # blocked_by[x] = set of pairs (a,b) such that x in pair_witness[(a,b)]
    blocked_by = [set() for _ in range(n + 1)]
    for (a, b), witnesses in pair_witness.items():
        for z in witnesses:
            if a != z and b != z:
                blocked_by[z].add(frozenset([a, b]))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        lst = sorted(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if frozenset([a, b]) in blocked_by[x]:
                    return False
        return True
    
    # Branch and bound with Tomita-style coloring bound
    # We'll use a recursive approach with memoization-like pruning
    
    # First, let's get a good initial solution using the existing heuristic
    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(initial_set, max_iters=100):
        """Local search to improve initial solution."""
        current_set = set(initial_set)
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 5 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to add an element
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap
            for out_elem in list(current_set):
                if time.time() > deadline - 5:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    # Get initial solution
    random.seed(42)
    best_set = set()
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = len(blocked_by[x])
    
    for i in range(10):
        if time.time() > deadline - 10:
            break
        if i % 2 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        else:
            order = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
        s = greedy_add(order)
        s_improved = local_search(s)
        if len(s_improved) > len(best_set):
            best_set = s_improved
    
    # Now use branch and bound to try to find a better solution
    # We'll use a different approach: since the problem size is manageable (128 elements),
    # we can try an exact branch and bound with good pruning.
    
    # Actually, for this specific problem structure, let me try a different exact approach:
    # Use the fact that we can formulate this as a maximum independent set in a 
    # 3-uniform hypergraph, and use branch and bound with the following pruning:
    # 
    # Upper bound: |S_current| + (n - |S_current|) but this is trivial.
    # Better: use the conflict structure to bound.
    
    # Let me try a simpler but effective approach: iterative improvement with 
    # more aggressive local search and multiple random restarts.
    
    # Actually, let me implement a proper branch and bound for this problem.
    # The key is to order vertices well and use a tight upper bound.
    
    # For now, let's enhance the local search significantly
    best_set = set(best_set)
    
    # More aggressive local search with multiple passes
    for pass_num in range(20):
        if time.time() > deadline - 3:
            break
        
        improved = False
        # Try all possible single additions
        for in_elem in range(1, n + 1):
            if in_elem in best_set:
                continue
            if can_add(in_elem, best_set):
                best_set.add(in_elem)
                improved = True
        
        if not improved:
            # Try 1-for-1 swaps
            for out_elem in list(best_set):
                if time.time() > deadline - 3:
                    break
                trial_base = best_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in best_set:
                        continue
                    if can_add(in_elem, trial_base):
                        best_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try 2-for-1 swaps (remove 2, add 1) - this can sometimes help
            found_improvement = False
            elements_list = list(best_set)
            for i in range(len(elements_list)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(elements_list)):
                    if time.time() > deadline - 3:
                        break
                    out_pair = {elements_list[i], elements_list[j]}
                    trial_base = best_set - out_pair
                    for in_elem in range(1, n + 1):
                        if in_elem in best_set:
                            continue
                        if can_add(in_elem, trial_base):
                            new_set = trial_base | {in_elem}
                            # This doesn't increase size, so skip unless we want to enable further improvements
                            # Actually, 2-for-1 decreases size by 1, so skip
                            pass
                    # Skip 2-for-1 as it reduces cardinality
                # No need to continue if no improvement found in this pass
            break
    
    # Final verification
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
