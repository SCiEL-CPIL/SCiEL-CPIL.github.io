"""Maximize reciprocal weight of 3-AP-free subset of [1,256] using branch-and-bound."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

# Precompute all 3-AP conflicts
# For each x, store the set of pairs (a,b) such that {a,b,x} forms a 3-AP
# Actually, for efficiency, let's precompute for each x, which other elements conflict with it
# A 3-AP is (a, b, c) where a+b=2c, a<c<b, or a<b<c with b=(a+c)/2
# For a given set S, x conflicts if there exist a,b in S such that {a,b,x} is a 3-AP

# Precompute: for each x, the set of all y such that x and y are in some 3-AP
# More precisely: for each x, precompute which pairs (a,b) in [1,N] form a 3-AP with x

def precompute_conflicts():
    """For each x, compute the set of other elements that form a 3-AP with x and some third element."""
    # conflicts[x] = set of y such that there exists z where {x,y,z} is a 3-AP
    # But for branch-and-bound, we need: given a partial set S, can we add x?
    # x can be added iff for all a in S, the "partner" of a with x doesn't create an AP
    # Actually: x creates a 3-AP with S iff there exist a,b in S such that {a,b,x} is a 3-AP
    # This happens iff:
    #   - x is the middle: a + c = 2x for some a,c in S, a != c
    #   - x is the max: a + x = 2b for some a,b in S, a < b < x  
    #   - x is the min: x + b = 2a for some a,b in S, x < b < a
    
    # For efficient checking, precompute for each x:
    #   mid_partners[x] = set of pairs (a, 2x-a) where a != 2x-a and both in [1,N]
    #   max_partners[x] = set of pairs (a, (a+x)/2) where a < (a+x)/2 < x and (a+x) even
    #   min_partners[x] = set of pairs ((x+b)/2, b) where x < (x+b)/2 < b and (x+b) even
    
    # Simpler: for each x, precompute the set of all y such that x and y are in a 3-AP
    # i.e., there exists z with {x,y,z} a 3-AP
    # Then x conflicts with S iff there exists y in S such that y is in conflict_set[x] AND the third element is also in S
    # This is getting complicated. Let's just precompute all 3-APs.
    
    all_aps = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            c = 2*b - a
            if c > b and c <= N:
                all_aps.append((a, b, c))
    
    # For each x, store which other elements form a 3-AP with x
    # conflict_pairs[x] = set of (y, z) such that {x,y,z} is a 3-AP
    conflict_pairs = {x: set() for x in range(1, N+1)}
    for a, b, c in all_aps:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    return all_aps, conflict_pairs

def can_add(x, selected_set, conflict_pairs):
    """Check if x can be added to selected_set without creating a 3-AP."""
    for y, z in conflict_pairs[x]:
        if y in selected_set and z in selected_set:
            return False
    return True

def construct():
    start_time = time.time()
    
    # Precompute conflicts
    all_aps, conflict_pairs = precompute_conflicts()
    
    # Greedy initial solution
    greedy_set = set()
    for x in range(1, N+1):
        if can_add(x, greedy_set, conflict_pairs):
            greedy_set.add(x)
    
    def reciprocal_sum(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total
    
    best_set = set(greedy_set)
    best_score = reciprocal_sum(best_set)
    
    # Branch and bound
    # We process elements in order 1 to 256
    # At each step, we decide to include or exclude the next element
    # Upper bound: current_score + sum of 1/x for all remaining elements that could potentially be added
    
    # Sort elements by weight (1/x) descending for better pruning
    # But we need to be careful about the ordering for the conflict structure
    
    # Let's use a different approach: iterative deepening or simple B&B
    # State: (index, current_set, current_score)
    # We go through elements 1..256, at each step choose include or exclude
    
    # For upper bound: current_score + sum(1/x for x in remaining elements)
    # This is a loose bound but easy to compute
    
    # Precompute suffix sums of reciprocals
    suffix_sum = [Fraction(0, 1)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + Fraction(1, i)
    
    def branch_and_bound(idx, current_set, current_score):
        nonlocal best_set, best_score, start_time
        
        if time.time() - start_time > TIME_LIMIT - 1:
            return
        
        # Upper bound
        upper_bound = current_score + suffix_sum[idx]
        if upper_bound <= best_score:
            return
        
        if idx > N:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        x = idx
        
        # Option 1: Include x
        if can_add(x, current_set, conflict_pairs):
            new_set = current_set | {x}
            new_score = current_score + Fraction(1, x)
            branch_and_bound(idx + 1, new_set, new_score)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, current_set, current_score)
    
    # Start branch and bound
    branch_and_bound(1, set(), Fraction(0, 1))
    
    # If B&B didn't improve, use local search on greedy
    if best_score == reciprocal_sum(greedy_set) and time.time() - start_time < TIME_LIMIT - 5:
        # Simple local search: try swaps
        current = set(greedy_set)
        current_score = reciprocal_sum(current)
        
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 3:
            improved = False
            for remove_x in sorted(current):
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                reduced = current - {remove_x}
                reduced_score = current_score - Fraction(1, remove_x)
                
                for add_x in range(1, N+1):
                    if add_x in reduced:
                        continue
                    if can_add(add_x, reduced, conflict_pairs):
                        new_score = reduced_score + Fraction(1, add_x)
                        if new_score > current_score:
                            current = reduced | {add_x}
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current)
                            break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
