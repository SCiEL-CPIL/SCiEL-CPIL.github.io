"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85.0
    random.seed(42)
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute all forbidden triples
    # A triple (a,b,c) is forbidden if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For efficiency: for each possible LCM value L, find all pairs with that LCM,
    # then find triangles in the "pairwise LCM = L" graph
    forbidden = set()
    
    # For each L from 1 to 128*128 (max possible LCM), but actually LCM of two numbers
    # in [1,128] is at most 128*127 = 16256, but let's be smarter.
    # Group numbers by their divisors.
    
    # Better approach: for each pair (a,b), compute L = lcm(a,b).
    # Then for each c that is a divisor of L with lcm(a,c)=L and lcm(b,c)=L,
    # (a,b,c) is a forbidden triple.
    
    # Let's precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            lcm_table[(i,j)] = lcm(i,j)
    
    # For each L, collect all pairs with lcm = L
    from collections import defaultdict
    pairs_by_lcm = defaultdict(list)
    for (a,b), L in lcm_table.items():
        pairs_by_lcm[L].append((a,b))
    
    # For each L, find all c such that c divides L, and for each pair (a,b) with lcm=L,
    # check if lcm(a,c)=L and lcm(b,c)=L
    # Actually, for a fixed L and a pair (a,b) with lcm(a,b)=L,
    # c must satisfy: c | L, lcm(a,c)=L, lcm(b,c)=L
    # This means c must contain all prime power factors of L not in a,
    # and all prime power factors of L not in b.
    
    # Simpler: just iterate over all triples. With 128 elements, C(128,3) = 341,376 triples.
    # That's manageable.
    forbidden = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm(a, b)
            for c in range(b+1, N+1):
                if lab == lcm(a, c) == lcm(b, c):
                    forbidden.add((a, b, c))
    
    # Convert to index-based for speed
    # Index: 0..127 corresponds to 1..128
    forbidden_idx = set()
    for (a,b,c) in forbidden:
        forbidden_idx.add((a-1, b-1, c-1))
    
    n = N
    n_forbidden = len(forbidden_idx)
    
    # WalkSAT-style local search for Max-SAT
    # Variables: x[i] = True means element i+1 is selected
    # Clauses: for each forbidden triple (i,j,k), clause is (NOT x[i] OR NOT x[j] OR NOT x[k])
    # We want to maximize number of True variables subject to all clauses being satisfied.
    
    # Represent clauses as lists of (var, sign) where sign=1 means positive (var must be False for clause to be satisfied... wait)
    # Clause (NOT x[i] OR NOT x[j] OR NOT x[k]) is satisfied if at least one of x[i], x[j], x[k] is False.
    # So the clause is violated if all three are True.
    
    # For WalkSAT:
    # - State: assignment of n booleans
    # - A clause is unsatisfied if all its literals are False (i.e., all three variables are True)
    # - We want to minimize unsatisfied clauses while maximizing total True count.
    
    # Actually, let's think of it as: we want to maximize sum(x[i]) subject to all forbidden triples not all being True.
    # This is a maximum independent set in a 3-uniform hypergraph, or equivalently Max-SAT with negative 3-clauses.
    
    # WalkSAT approach:
    # Start with a random or greedy assignment.
    # While unsatisfied clauses exist:
    #   Pick a random unsatisfied clause.
    #   With probability p, flip a random variable in the clause.
    #   With probability 1-p, flip the variable in the clause that maximizes satisfied clauses (or minimizes unsatisfied).
    # Also periodically try to set more variables to True if it doesn't violate constraints.
    
    # Better: use a hybrid approach.
    # 1. Start with greedy: add elements in some order, skipping if it creates a forbidden triple.
    # 2. Then use local search: try to add removed elements, or swap elements.
    
    # Let's do:
    # Phase 1: Greedy with multiple orderings
    # Phase 2: Local search (add/remove/swap)
    # Phase 3: WalkSAT-style to escape local optima
    
    def is_valid(assignment):
        """Check if assignment (set of indices) has no forbidden triple."""
        s = set(assignment)
        for (i,j,k) in forbidden_idx:
            if i in s and j in s and k in s:
                return False
        return True
    
    def count_conflicts(assignment):
        """Count number of forbidden triples fully contained in assignment."""
        s = set(assignment)
        cnt = 0
        for (i,j,k) in forbidden_idx:
            if i in s and j in s and k in s:
                cnt += 1
        return cnt
    
    # Precompute for each index, which forbidden triples it participates in
    triples_by_var = defaultdict(list)
    for (i,j,k) in forbidden_idx:
        triples_by_var[i].append((j,k))
        triples_by_var[j].append((i,k))
        triples_by_var[k].append((i,j))
    
    def greedy(order):
        """Greedy: add elements in given order if no conflict."""
        selected = set()
        for idx in order:
            # Check if adding idx creates any forbidden triple
            ok = True
            for (j,k) in triples_by_var[idx]:
                if j in selected and k in selected:
                    ok = False
                    break
            if ok:
                selected.add(idx)
        return selected
    
    def local_search(selected, max_iters=50000):
        """Try to improve by adding, removing, or swapping elements."""
        selected = set(selected)
        removed = set(range(n)) - selected
        best_size = len(selected)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to add a removed element
            if removed:
                # Try elements in random order
                add_candidates = list(removed)
                random.shuffle(add_candidates)
                for idx in add_candidates:
                    if len(selected) + 1 <= best_size:  # Only add if it improves
                        # Check if adding idx is valid
                        ok = True
                        for (j,k) in triples_by_var[idx]:
                            if j in selected and k in selected:
                                ok = False
                                break
                        if ok:
                            selected.add(idx)
                            removed.discard(idx)
                            best_size = len(selected)
                            improved = True
                            break
            
            # Try to swap: remove one from selected, add one from removed
            if not improved and removed:
                # Try a few random swaps
                sel_list = list(selected)
                rem_list = list(removed)
                random.shuffle(sel_list)
                random.shuffle(rem_list)
                
                for rem_idx in rem_list[:50]:
                    for sel_idx in sel_list[:50]:
                        if rem_idx == sel_idx:
                            continue
                        # Try swap
                        new_selected = selected - {sel_idx} | {rem_idx}
                        # Check validity
                        ok = True
                        s = set(new_selected)
                        for (i,j,k) in forbidden_idx:
                            if i in s and j in s and k in s:
                                ok = False
                                break
                        if ok and len(new_selected) > len(selected):
                            selected = new_selected
                            removed = set(range(n)) - selected
                            best_size = len(selected)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return selected
    
    # Phase 1: Greedy with multiple orderings
    best_set = set()
    
    orders = [
        list(range(n)),           # 1,2,...,128
        list(range(n-1, -1, -1)), # 128,127,...,1
        sorted(range(n), key=lambda x: x+1, reverse=True),
    ]
    
    # Also try ordering by "difficulty" - elements that are in more forbidden triples first
    # or last
    triple_count = [len(triples_by_var[i]) for i in range(n)]
    orders.append(sorted(range(n), key=lambda x: triple_count[x]))  # easy first
    orders.append(sorted(range(n), key=lambda x: -triple_count[x])) # hard first
    
    # Random orders
    for _ in range(20):
        o = list(range(n))
        random.shuffle(o)
        orders.append(o)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        s = greedy(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Phase 2: Local search from best greedy result
    if time.time() < deadline - 5:
        improved = local_search(best_set)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Phase 3: WalkSAT-style search
    # Start from best_set, try to escape local optima
    if time.time() < deadline - 3:
        current = set(best_set)
        best_found = set(best_set)
        best_size = len(best_set)
        
        p_flip = 0.3
        stall_count = 0
        
        while time.time() < deadline - 2:
            # Count unsatisfied clauses (forbidden triples fully in current)
            conflicts = []
            for (i,j,k) in forbidden_idx:
                if i in current and j in current and k in current:
                    conflicts.append((i,j,k))
            
            if not conflicts:
                # Valid solution
                if len(current) > best_size:
                    best_size = len(current)
                    best_found = set(current)
                # Try to add more
                added = False
                for idx in range(n):
                    if idx not in current:
                        ok = True
                        for (j,k) in triples_by_var[idx]:
                            if j in current and k in current:
                                ok = False
                                break
                        if ok:
                            current.add(idx)
                            added = True
                            break
                if not added:
                    break
            else:
                # Pick a random conflict
                tri = random.choice(conflicts)
                # With probability p, flip random var in tri
                # With probability 1-p, flip var that minimizes conflicts
                if random.random() < p_flip:
                    var = random.choice(tri)
                else:
                    # For each var in tri, count how many conflicts would be resolved vs created
                    best_var = tri[0]
                    best_score = -1
                    for var in tri:
                        # If we remove var, how many conflicts go away?
                        removed_conflicts = 0
                        for c in conflicts:
                            if var in c:
                                removed_conflicts += 1
                        # New conflicts created by removing var: none (removing can't create)
                        # But we might want to add it back later
                        score = removed_conflicts
                        if score > best_score:
                            best_score = score
                            best_var = var
                    var = best_var
                
                if var in current:
                    current.discard(var)
                else:
                    current.add(var)
            
            if len(current) > best_size:
                best_size = len(current)
                best_found = set(current)
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 10000:
                break
    
    best_set = best_found
    
    # Final: convert to list of values (1-indexed)
    result = sorted([i+1 for i in best_set])
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
