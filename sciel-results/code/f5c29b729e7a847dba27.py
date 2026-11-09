import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # 85s safety margin (evaluator kills at 90s)
    
    n = 192
    max_k = 48
    
    # Precompute conflict structure: for each a, list of triples (b,c,d) with b<c<d, a not in {b,c,d}, where a | (b+c+d)
    # This allows fast checking: when we have a set S, a conflicts if any of its conflict-triples is a subset of S\{a}
    
    # For efficiency, precompute for each a the set of "conflict triples" as frozensets
    conflict_triples = {}
    for a in range(1, n+1):
        triples = []
        for triple in combinations(range(1, n+1), 3):
            if a in triple:
                continue
            if sum(triple) % a == 0:
                triples.append(frozenset(triple))
        conflict_triples[a] = triples
    
    # For faster checking, also precompute: for each pair (b,c), which a's have a conflict triple containing {b,c}?
    # Actually, let's use a different approach: when checking if x can be added to set S,
    # we need to check:
    # 1. For x itself: does x divide sum of any 3 elements in S?
    # 2. For each a in S: does a divide sum of any 3 elements in S\{a} that includes x?
    
    def check_addition(selected, x):
        """Check if adding x to selected keeps the property valid. Returns True if valid."""
        sel = list(selected)
        
        # Check 1: x should not divide sum of any 3 distinct elements in sel
        for triple in combinations(sel, 3):
            if sum(triple) % x == 0:
                return False
        
        # Check 2: for each a in sel, a should not divide sum of any 3 distinct elements in (sel - {a}) ∪ {x}
        # The new triples involving x are: (x, b, c) where b, c are in sel - {a}
        # So we need: for each a in sel, for each pair (b, c) in combinations(sel - {a}, 2), check if a | (x + b + c)
        for i, a in enumerate(sel):
            others_without_a = [sel[j] for j in range(len(sel)) if j != i]
            for pair in combinations(others_without_a, 2):
                s = x + pair[0] + pair[1]
                if s % a == 0:
                    return False
        
        return True
    
    def full_check(selected):
        """Verify the entire set is valid."""
        for i, a in enumerate(selected):
            others = [selected[j] for j in range(len(selected)) if j != i]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    # Greedy: build a set by adding elements in a given order if valid
    def greedy_build(order):
        """Build a set by adding elements in the given order if valid."""
        selected = []
        for x in order:
            if len(selected) >= max_k:
                break
            if check_addition(selected, x):
                selected.append(x)
        return selected
    
    # Degree-based greedy: at each step, pick the candidate with minimum "conflict degree"
    # Conflict degree of x given current set S = number of constraints that would be violated if x were added
    # Approximate: for each a in S, count how many pairs (b,c) in S\{a} have a | (x+b+c)
    # Plus: number of triples in S where x | sum(triple)
    
    def compute_conflict_degree(x, selected):
        """Compute approximate conflict degree of x given current set."""
        degree = 0
        sel = list(selected)
        
        # Count triples in sel where x divides their sum
        for triple in combinations(sel, 3):
            if sum(triple) % x == 0:
                degree += 10  # weight heavily
        
        # For each a in sel, count pairs (b,c) in sel\{a} where a | (x+b+c)
        for i, a in enumerate(sel):
            others = [sel[j] for j in range(len(sel)) if j != i]
            for pair in combinations(others, 2):
                if (x + pair[0] + pair[1]) % a == 0:
                    degree += 1
        
        return degree
    
    def degree_greedy():
        """Greedy that at each step picks the unused element with minimum conflict degree."""
        selected = []
        used = set()
        candidates = list(range(1, n+1))
        
        while len(selected) < max_k and candidates:
            if time.time() > deadline - 5:
                break
            
            # Find candidate with minimum conflict degree
            best_x = None
            best_degree = float('inf')
            for x in candidates:
                d = compute_conflict_degree(x, selected)
                if d < best_degree:
                    best_degree = d
                    best_x = x
                elif d == best_degree and best_x is not None:
                    # Tie-break: prefer larger numbers (less likely to divide small sums)
                    if x > best_x:
                        best_x = x
            
            if best_x is None:
                break
            
            if check_addition(selected, best_x):
                selected.append(best_x)
                used.add(best_x)
                candidates.remove(best_x)
            else:
                # Shouldn't happen if degree is 0, but handle gracefully
                candidates.remove(best_x)
        
        return selected
    
    # Try multiple strategies
    best = []
    
    # Strategy 1: largest first
    order1 = list(range(n, 0, -1))
    sol1 = greedy_build(order1)
    if len(sol1) > len(best):
        best = sol1
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: degree-based greedy
    sol2 = degree_greedy()
    if len(sol2) > len(best):
        best = sol2
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: randomized greedy with restarts
    random.seed(42)
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random order with bias toward large numbers
        order3 = sorted(range(1, n+1), key=lambda x: random.random() * 30 + x * 0.5)
        sol3 = greedy_build(order3)
        if len(sol3) > len(best):
            best = sol3
    
    if time.time() > deadline:
        return best
    
    # Local search: 1-add and 2-swap moves
    def local_search(selected, time_limit=None):
        """Improve set via 1-add and 2-swap moves."""
        if time_limit is None:
            time_limit = deadline - 5
        
        selected = list(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-add: try adding any unused element
            used = set(selected)
            for x in range(n, 0, -1):
                if x not in used:
                    if check_addition(selected, x):
                        selected.append(x)
                        improved = True
                        break
            
            if len(selected) >= max_k:
                break
            
            # 2-swap: try removing one element and adding another
            for i in range(len(selected)):
                if time.time() > time_limit:
                    break
                temp = selected[:i] + selected[i+1:]
                used_temp = set(temp)
                for x in range(n, 0, -1):
                    if x not in used_temp:
                        if check_addition(temp, x):
                            new_set = temp + [x]
                            if len(new_set) > len(selected):
                                selected = new_set
                                improved = True
                                break
                if improved:
                    break
        
        return selected
    
    best = local_search(best)
    
    if time.time() > deadline:
        return best
    
    # More aggressive: random restarts from perturbed best
    random.seed(123)
    for attempt in range(15):
        if time.time() > deadline - 8:
            break
        
        if len(best) > 2:
            temp = list(best)
            # Remove 1-2 random elements
            k_remove = random.randint(1, min(2, len(temp)))
            for _ in range(k_remove):
                idx = random.randint(0, len(temp)-1)
                temp.pop(idx)
            # Try to rebuild
            temp = local_search(temp, time_limit=deadline - 3)
            if len(temp) > len(best):
                best = temp
    
    return best
