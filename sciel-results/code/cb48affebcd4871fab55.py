"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    N = 128
    all_numbers = list(range(1, N + 1))
    
    # Precompute LCMs for speed
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_table[(a, b)] = lcm(a, b)
            
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in combinations(all_numbers, 3):
        if lcm_table[(a, b)] == lcm_table[(a, c)] == lcm_table[(b, c)]:
            bad_triples.append((a, b, c))
            
    if not bad_triples:
        return all_numbers
        
    # Map number to list of triple indices it belongs to
    num_to_triples = {x: [] for x in all_numbers}
    for i, t in enumerate(bad_triples):
        for x in t:
            num_to_triples[x].append(i)
    
    # Build conflict graph (2-shadow): edge between x,y if they share a bad triple
    conflict = {x: set() for x in all_numbers}
    for t in bad_triples:
        a, b, c = t
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)
    
    # Precompute for each x, the list of (u, v) pairs such that {x,u,v} is bad
    x_to_pairs = {x: [] for x in all_numbers}
    for t in bad_triples:
        a, b, c = t
        x_to_pairs[a].append((b, c))
        x_to_pairs[b].append((a, c))
        x_to_pairs[c].append((a, b))
    
    best_set = set()
    best_size = 0
    
    def is_valid(S):
        """Check if S contains no bad triple."""
        S_set = set(S) if not isinstance(S, set) else S
        for t in bad_triples:
            if t[0] in S_set and t[1] in S_set and t[2] in S_set:
                return False
        return True
    
    def try_add_elements(S, order=None):
        """Try to add elements back to S in given order (or random if None)."""
        S = set(S)
        candidates = [x for x in all_numbers if x not in S]
        if order is not None:
            candidates.sort(key=lambda x: order.index(x) if x in order else len(order))
        else:
            random.shuffle(candidates)
        
        for x in candidates:
            can_add = True
            for u, v in x_to_pairs[x]:
                if u in S and v in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S
    
    def local_search_polish(S, max_rounds=20):
        """Try to improve S by removing one element and adding two, or similar moves."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            # Try removing each element and seeing if we can add back more
            for x in list(S):
                S_test = S - {x}
                # Try to add elements that were removed
                added = 0
                for y in all_numbers:
                    if y not in S_test:
                        can_add = True
                        for u, v in x_to_pairs[y]:
                            if u in S_test and v in S_test:
                                can_add = False
                                break
                        if can_add:
                            S_test.add(y)
                            added += 1
                if len(S_test) > len(S):
                    S = S_test
                    improved = True
                    break
            # Also try single-swap: remove x, add y
            for x in list(S):
                for y in all_numbers:
                    if y not in S:
                        S_test = (S - {x}) | {y}
                        if is_valid(S_test):
                            # Check if we can add more
                            for z in all_numbers:
                                if z not in S_test:
                                    can_add = True
                                    for u, v in x_to_pairs[z]:
                                        if u in S_test and v in S_test:
                                            can_add = False
                                            break
                                    if can_add:
                                        S_test.add(z)
                            if len(S_test) > len(S):
                                S = S_test
                                improved = True
                                break
                if improved:
                    break
        return S
    
    # Branch and bound for Maximum Independent Set on hypergraph
    def branch_and_bound(available, current_set, depth=0):
        """
        available: set of numbers not yet decided
        current_set: set of numbers chosen so far
        Returns best independent set found.
        """
        nonlocal best_set, best_size
        
        if time.time() - start_time > timeout:
            return
        
        # Upper bound: |current_set| + |available|
        # Tighter bound using conflict graph: greedy vertex cover on induced subgraph
        if len(current_set) + len(available) <= best_size:
            return
        
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Choose a bad triple that has all elements in available
        # If none, we can add all remaining
        active_triples = []
        for t in bad_triples:
            if t[0] in available and t[1] in available and t[2] in available:
                active_triples.append(t)
        
        if not active_triples:
            # No conflicts among available, add all
            new_set = current_set | available
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = set(new_set)
            return
        
        # Pick a triple with minimum "degree" in available
        # Use the triple where the sum of degrees of its elements in the conflict graph (restricted to available) is minimized
        def triple_score(t):
            score = 0
            for x in t:
                # Count neighbors of x in available
                score += len(conflict[x] & available)
            return score
        
        active_triples.sort(key=triple_score)
        t = active_triples[0]
        a, b, c = t
        
        # Branch: exclude a, or exclude b, or exclude c
        # To ensure we don't include all three, we must exclude at least one.
        # We branch on excluding each of a, b, c.
        
        # Heuristic: exclude the one with highest degree first (most likely to be excluded in optimal)
        degrees = [(len(conflict[x] & available), x) for x in [a, b, c]]
        degrees.sort(key=lambda x: -x[0])
        
        for _, x_to_exclude in degrees:
            new_available = available - {x_to_exclude}
            branch_and_bound(new_available, current_set, depth + 1)
            if time.time() - start_time > timeout:
                return
        
        # Also branch on including a (and thus excluding b and c)
        # This is covered by the above branches since we must exclude at least one.
        # Actually, the above branches cover: exclude a, exclude b, exclude c.
        # But we also need to consider including a, which means excluding b and c.
        # That's covered by "exclude b" branch (where a might still be included).
        # Wait, no. The branches are:
        # 1. Exclude a: new_available = available - {a}
        # 2. Exclude b: new_available = available - {b}
        # 3. Exclude c: new_available = available - {c}
        # In branch 1, a is not in available, so it won't be included.
        # In branch 2, b is not in available, but a and c could both be included if no other conflicts.
        # But {a,b,c} is a bad triple, so we can't include all three.
        # In branch 2, b is excluded, so {a,b,c} is broken. a and c can both be included.
        # So the three branches cover all cases where not all of {a,b,c} are included.
        # This is correct.
        
    # Greedy initial solution
    def greedy_solution():
        S = set(all_numbers)
        active_triples = set(range(len(bad_triples)))
        deg = {x: len(num_to_triples[x]) for x in all_numbers}
        
        while active_triples:
            t_idx = random.choice(list(active_triples))
            a, b, c = bad_triples[t_idx]
            
            if a in S and b in S and c in S:
                candidates = [(deg[a], a), (deg[b], b), (deg[c], c)]
                candidates.sort(key=lambda x: -x[0])
                max_deg = candidates[0][0]
                top_candidates = [x for d, x in candidates if d == max_deg]
                to_remove = random.choice(top_candidates)
                
                S.remove(to_remove)
                
                for t_idx2 in num_to_triples[to_remove]:
                    if t_idx2 in active_triples:
                        active_triples.remove(t_idx2)
                        t = bad_triples[t_idx2]
                        for y in t:
                            if y != to_remove:
                                deg[y] -= 1
        
        return S
    
    # Run multiple restarts with different strategies
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Strategy 1: Greedy removal + polish
        if restart % 3 == 0:
            S = greedy_solution()
            S = try_add_elements(S)
            S = local_search_polish(S, max_rounds=5)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        
        # Strategy 2: Randomized greedy with different ordering
        elif restart % 3 == 1:
            S = set(all_numbers)
            active_triples = set(range(len(bad_triples)))
            deg = {x: len(num_to_triples[x]) for x in all_numbers}
            
            while active_triples:
                t_idx = random.choice(list(active_triples))
                a, b, c = bad_triples[t_idx]
                
                if a in S and b in S and c in S:
                    # Random choice among the three
                    to_remove = random.choice([a, b, c])
                    
                    S.remove(to_remove)
                    
                    for t_idx2 in num_to_triples[to_remove]:
                        if t_idx2 in active_triples:
                            active_triples.remove(t_idx2)
                            t = bad_triples[t_idx2]
                            for y in t:
                                if y != to_remove:
                                    deg[y] -= 1
            
            S = try_add_elements(S)
            S = local_search_polish(S, max_rounds=5)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        
        # Strategy 3: Branch and bound
        else:
            # Start with a good initial solution from greedy
            S_init = greedy_solution()
            S_init = try_add_elements(S_init)
            if len(S_init) > best_size:
                best_size = len(S_init)
                best_set = set(S_init)
            
            # Use branch and bound to try to improve
            # Start with available = all numbers, current_set = empty
            # But this is too large. Instead, use the greedy solution as a lower bound
            # and try to find a better one.
            
            # For B&B, we need to be careful about the search space.
            # Let's try a limited-depth B&B.
            
            # Actually, let's just do more local search on the current best
            S = local_search_polish(best_set, max_rounds=10)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
    
    # Final polish
    if best_set:
        best_set = try_add_elements(best_set)
        best_set = local_search_polish(best_set, max_rounds=20)
    
    return list(best_set) if best_set else [1]

# EVOLVE-BLOCK-END
