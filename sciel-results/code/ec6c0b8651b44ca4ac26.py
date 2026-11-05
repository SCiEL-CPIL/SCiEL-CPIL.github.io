import time
from math import gcd
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Strategy:
    1. Precompute the set of "bad" triples.
    2. Use a Branch and Bound (B&B) approach on the 3-uniform hypergraph.
    3. Represent the state as a bitmask for O(1) conflict checks where possible, 
       or efficient set operations.
    4. Use a strong upper bound: current_size + (number of remaining candidates 
       that don't immediately conflict with the current set, divided by a 
       heuristic factor for triple conflicts).
    5. Initial solution via fast greedy to set a high lower bound.
    """
    
    deadline = time.time() + 85.0
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We can optimize this: if gcd(a,b) == d, then gcd(a,c) must be d and gcd(b,c) must be d.
    # This implies a, b, c are all multiples of d, and a/d, b/d, c/d are pairwise coprime? 
    # No, gcd(a/d, b/d) must be 1? 
    # Let a = d*a', b = d*b'. gcd(a,b)=d => gcd(a',b')=1.
    # We need gcd(a,c)=d => gcd(a',c')=1.
    # We need gcd(b,c)=d => gcd(b',c')=1.
    # So a', b', c' are pairwise coprime.
    
    bad_triples = []
    # For efficiency, we can iterate over possible gcds d.
    # If gcd(a,b,c) is not 1, say d, then a=d*a', etc.
    # The condition is that a', b', c' are pairwise coprime.
    # This structure allows for faster generation, but for N=128, brute force 
    # with optimized gcd checks is likely fast enough for precomputation.
    
    # Precompute gcds for all pairs to speed up triple checking
    gcds = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcds[i][j]
            
    # Identify bad triples
    # To make B&B fast, we need to know for any pair (a,b) in the set, 
    # which c's would create a bad triple.
    # Let's precompute for each pair (a,b), the set of c's such that (a,b,c) is bad.
    # conflict_map[(a,b)] = set of c
    conflict_map = {}
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcds[a][b]
            # If g_ab == 1, can it be a bad triple?
            # gcd(a,c)=1, gcd(b,c)=1. Yes, if c is coprime to both.
            # So (a,b,c) with gcds (1,1,1) is bad.
            # We need to find c such that gcd(a,c)==g_ab and gcd(b,c)==g_ab.
            
            candidates = []
            for c in range(1, N+1):
                if c == a or c == b:
                    continue
                if gcds[a][c] == g_ab and gcds[b][c] == g_ab:
                    candidates.append(c)
            if candidates:
                conflict_map[(a,b)] = candidates
                
    # Also need the reverse: for a single element c, which pairs (a,b) conflict with it?
    # Or better, for the B&B, we maintain the set of "available" candidates.
    # When we add x to the set, we must remove any y such that there exists z in the set 
    # where (x,y,z) is bad.
    # Actually, the constraint is: No 3 distinct elements.
    # So if we have set S, we can add x if there are no a,b in S such that (x,a,b) is bad.
    
    # Precompute: for each x, and for each pair (a,b) in conflict_map, 
    # we know that if a,b are in S, x cannot be added.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Data structure for fast checking:
    # For each x, store a list of (a,b) pairs that form a bad triple with x.
    # x_conflicts[x] = list of (a,b)
    x_conflicts = {x: [] for x in range(1, N+1)}
    for (a,b), cs in conflict_map.items():
        for c in cs:
            # Add (a,b) to c's conflicts
            x_conflicts[c].append((a,b))
            # Add (a,c) to b's conflicts? No, (a,b,c) is the triple.
            # If we pick c, we conflict with the pair (a,b).
            # If we pick b, we conflict with the pair (a,c).
            # If we pick a, we conflict with the pair (b,c).
            # Let's be consistent. 
            # x_conflicts[x] contains pairs (y,z) such that {x,y,z} is a bad triple.
            pass
            
    # Re-do x_conflicts properly
    x_conflicts = {x: [] for x in range(1, N+1)}
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcds[a][b]
            for c in range(1, N+1):
                if c == a or c == b: continue
                if gcds[a][c] == g_ab and gcds[b][c] == g_ab:
                    # Triple (a,b,c) is bad
                    x_conflicts[a].append((b,c))
                    x_conflicts[b].append((a,c))
                    x_conflicts[c].append((a,b))

    def is_conflict(x, selected_set):
        """Check if x can be added to selected_set."""
        # x can be added if no pair (y,z) in selected_set is in x_conflicts[x]
        # To make this fast, we can check if any y in x_conflicts[x][0] is in selected_set,
        # and if so, check if the corresponding z is also in selected_set.
        for y, z in x_conflicts[x]:
            if y in selected_set and z in selected_set:
                return False
        return True

    def greedy_initial():
        """Get a good initial solution using greedy."""
        selected = set()
        # Sort by degree (number of conflicts) ascending? 
        # Or just random shuffle and add if valid.
        # Let's try multiple random orders.
        best_size = 0
        best_set = set()
        
        for _ in range(50):
            random.shuffle(numbers)
            s = set()
            for x in numbers:
                if is_conflict(x, s):
                    s.add(x)
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)
                
        return best_set

    # Get initial lower bound
    current_best_set = greedy_initial()
    best_size = len(current_best_set)
    
    # Branch and Bound
    # State: 
    #   selected: set of chosen numbers
    #   candidates: set of numbers that can still be chosen (subset of remaining)
    #   upper_bound: theoretical max size from this state
    
    # To make B&B fast, we need a strong upper bound.
    # Simple bound: len(selected) + len(candidates)
    # Better bound: If we have candidates C, the max independent set in the hypergraph induced by C 
    # is at most |C|. But we can prune if len(selected) + len(candidates) <= best_size.
    
    # We can also use a packing bound: 
    # Each selected element "uses up" its conflicts.
    # A simple heuristic bound: len(selected) + sum(1 for c in candidates if is_conflict(c, selected))
    # This is just the greedy extension size. If even the greedy extension from this state 
    # doesn't beat best_size, we can prune. This is expensive to compute every node.
    
    # Alternative: Use a bitmask if N is small enough? 128 bits fits in 2 Python ints or 4 32-bit ints.
    # But Python sets are fine for N=128 if we are careful.
    
    # Let's use a recursive search with ordering.
    # Sort candidates by "degree" in the remaining hypergraph to prune earlier.
    
    def get_degree(x, candidates):
        """Count how many bad triples x is in with other candidates."""
        count = 0
        for y, z in x_conflicts[x]:
            if y in candidates and z in candidates:
                count += 1
        return count

    def bnb(selected, candidates, depth=0):
        nonlocal best_size, current_best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Pruning 1: Upper bound
        # Even if we pick all candidates, we can't exceed len(selected) + len(candidates)
        if len(selected) + len(candidates) <= best_size:
            return
            
        if not candidates:
            if len(selected) > best_size:
                best_size = len(selected)
                current_best_set = set(selected)
            return

        # Heuristic Upper Bound:
        # Try a fast greedy completion to see if it can beat best_size
        # This is O(N^2) per node, might be too slow.
        # Let's stick to the simple bound first, and add a "greedy check" if we have time.
        
        # Order candidates: Pick the one with highest degree in the subgraph? 
        # Or lowest? 
        # Usually, picking the most constrained first helps prune.
        # Let's sort candidates by degree in the remaining hypergraph (descending).
        
        # To avoid sorting every time, we can just pick one. 
        # But for B&B, ordering matters.
        
        # Optimization: If we have time, do a quick greedy check.
        # Greedy check:
        #   temp_selected = set(selected)
        #   temp_candidates = set(candidates)
        #   while temp_candidates:
        #       pick x with min degree?
        #       if is_conflict(x, temp_selected): temp_selected.add(x)
        #       temp_candidates.remove(x)
        #   if len(temp_selected) <= best_size: return
        
        # Let's implement a faster bound:
        # The maximum size of an independent set in a 3-uniform hypergraph is bounded by 
        # |V| / (min_degree + 1)? No, that's for graphs.
        
        # Let's just use the simple bound and rely on the initial greedy being strong.
        # To improve, we can try adding the "greedy check" only at shallow depths.
        
        if depth < 10:
            # Greedy completion check
            temp_sel = set(selected)
            temp_cand = list(candidates)
            # Sort by degree in current subgraph
            # This is expensive. Let's just shuffle and add? 
            # Or sort by global degree?
            # Let's just do a simple greedy: add if valid.
            # We want to see if we CAN beat best_size.
            # If a random greedy doesn't, it's likely no solution does.
            # But this is probabilistic.
            
            # Let's try a deterministic greedy: 
            # Pick candidates in order of increasing "conflict count" with current selected?
            # Actually, just checking if len(selected) + len(candidates) > best_size is necessary.
            # Let's add a stronger bound: 
            # If the number of candidates that are "safe" to add immediately is small, 
            # and the rest conflict, we might prune.
            
            safe_count = 0
            for c in candidates:
                if is_conflict(c, selected):
                    safe_count += 1
            
            # Even if we add all safe ones, they might conflict with each other.
            # Upper bound: len(selected) + safe_count
            if len(selected) + safe_count <= best_size:
                return

        # Choose next element to branch on
        # Heuristic: Pick the element with the highest degree in the remaining hypergraph
        # to maximize pruning.
        
        # Calculate degrees for candidates
        # To save time, we can approximate or just pick one.
        # Let's pick the one with the highest number of conflicts with other candidates.
        
        best_x = None
        best_deg = -1
        
        # Limit the search for the best branch to speed up
        # If candidates is large, sampling might be better?
        # For N=128, calculating degree for all candidates is O(N^3) worst case?
        # x_conflicts[x] has size ~N. Checking membership in candidates is O(1) if set.
        # So O(|candidates| * |conflicts[x]|). 
        # |conflicts[x]| is roughly N^2/12? No, much smaller.
        # Let's just do it.
        
        for x in candidates:
            deg = 0
            for y, z in x_conflicts[x]:
                if y in candidates and z in candidates:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_x = x
                
        if best_x is None:
             # No candidates?
             if len(selected) > best_size:
                 best_size = len(selected)
                 current_best_set = set(selected)
             return

        # Branch 1: Include best_x
        # Check if best_x is valid with selected
        if is_conflict(best_x, selected):
            new_selected = selected | {best_x}
            # Remove from candidates any element that conflicts with best_x 
            # given the new selected set? 
            # Actually, if we add best_x, we must remove any c in candidates 
            # such that there exists y in selected where (c, y, best_x) is bad?
            # No, the constraint is "no three distinct".
            # If we have S, and add x, we must ensure no a,b in S form a bad triple with x.
            # We already checked is_conflict(x, S).
            # Now, for future additions, we must ensure they don't form bad triples 
            # with pairs in S U {x}.
            # So, the new "candidates" must be filtered:
            # A candidate c is valid if:
            # 1. It was valid before (no conflict with S)
            # 2. It has no conflict with x AND any element in S? 
            #    i.e., no (c, y, x) is bad for any y in S.
            #    This is equivalent to: is_conflict(c, S | {x}) is True?
            #    Wait, is_conflict(c, S) checks if c conflicts with any pair in S.
            #    We need c to not conflict with any pair in S U {x}.
            #    Pairs in S U {x} are:
            #    - Pairs in S (already handled by c being in candidates? 
            #      No, candidates were filtered based on S. So c is safe w.r.t S pairs.)
            #    - Pairs involving x and s in S.
            #    So we need to check if (c, x, s) is bad for any s in S.
            #    This means: for any s in S, if (c, x, s) is a bad triple, c is invalid.
            
            # Let's filter candidates:
            new_candidates = set()
            for c in candidates:
                if c == best_x:
                    continue
                # Check if c is still valid with S U {best_x}
                # We know c is valid with S.
                # We need to check if c conflicts with best_x given S.
                # i.e., is there s in S such that (c, best_x, s) is bad?
                valid = True
                # Iterate through conflicts of c
                # If (c, best_x, s) is bad, then (best_x, s) is in x_conflicts[c]
                # So check if best_x is in the first or second element of any pair in x_conflicts[c]
                # and the other element is in S.
                
                # Faster: Precomputed? 
                # Let's just check is_conflict(c, new_selected) directly?
                # is_conflict(c, new_selected) checks if there is a pair (y,z) in new_selected 
                # such that (c,y,z) is bad.
                # This includes pairs from S (which we know are safe) and pairs from {x} U S.
                # Since we know c is safe w.r.t S, we only need to check pairs involving x.
                # But is_conflict is O(|conflicts[c]|). 
                # Given |S| is small, this is fast.
                
                if is_conflict(c, new_selected):
                    new_candidates.add(c)
                    
            bnb(new_selected, new_candidates, depth + 1)

        # Branch 2: Exclude best_x
        new_candidates = candidates - {best_x}
        bnb(selected, new_candidates, depth + 1)

    # Run B&B
    # To make it faster, we can start with a smaller set of candidates?
    # The greedy initial solution is strong. 
    # Let's run B&B.
    
    # Note: B&B on 128 nodes might be too slow if the branching factor is high.
    # The "safe_count" bound helps.
    
    # Let's add a time check inside the loop? 
    # The recursion handles it.
    
    # One issue: The initial greedy might not be optimal. 
    # B&B will find the optimal if it completes.
    # If it doesn't complete, it returns the best found.
    
    # To ensure we don't time out, we rely on the deadline check in bnb.
    
    # Let's run it.
    # We pass the full set of numbers as candidates initially?
    # Or can we prune candidates that are "dominated"?
    # For now, start with all numbers.
    
    # Optimization: Remove numbers that are "too conflicting" with the initial set?
    # No, B&B explores both include and exclude.
    
    # Start B&B
    # We use the empty set as selected, and all numbers as candidates?
    # No, we can seed it with the greedy solution to get a better bound immediately?
    # If we seed with greedy, we are looking for improvements.
    # But B&B from empty set with a good lower bound is standard.
    
    # Let's use the greedy set as the lower bound, and start B&B from empty set.
    # This allows B&B to find a different, potentially larger set.
    
    bnb(set(), set(numbers), 0)
    
    if not current_best_set:
        current_best_set = {1}
        
    return sorted(current_best_set)

# EVOLVE-BLOCK-END
