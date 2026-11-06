"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute 3-APs: list of (a, b, c) where a < b < c and a+c=2b
    # We need to check if a subset contains any such triple.
    # For efficiency in local search, we can precompute for each x, the pairs (y, z) 
    # such that {x, y, z} forms a 3-AP.
    # Actually, for a fast penalty calculation, it's better to precompute 
    # for each x, the list of other elements that form a 3-AP with x.
    
    # Let's precompute a list of all 3-APs in [1, n]
    # There are not too many.
    # a, b, c in [1, n], a < b < c, a+c = 2b
    # b can be from 2 to n-1.
    # For a given b, a can be 1..b-1, c = 2b-a.
    # We need c <= n.
    
    ap_list = []
    for b in range(2, n):
        for a in range(1, b):
            c = 2*b - a
            if c <= n:
                ap_list.append((a, b, c))
                
    # To speed up penalty calculation, for each x, store the indices of APs involving x?
    # Or simply iterate all APs? 
    # Number of APs:
    # Sum_{b=2}^{n-1} min(b-1, n-b)
    # For n=256, this is roughly sum_{b=2}^{128} (b-1) + sum_{b=129}^{255} (256-b)
    # ~ 128^2/2 * 2 ~ 16000 APs.
    # Iterating 16000 APs per step is fast in Python if optimized?
    # 16000 * 50000 steps = 800 million ops. Might be too slow in pure Python.
    
    # Better: Precompute for each x, the set of pairs (y, z) such that x, y, z form an AP.
    # Or even better, for each x, precompute the list of (y, z) such that if x is in the set,
    # and y is in the set, then z must NOT be in the set.
    
    # Let's use a bitmask or boolean array for the set S.
    # s = [False] * (n+1)
    
    # Penalty function:
    # P(S) = sum_{(a,b,c) in APs} I(a in S) * I(b in S) * I(c in S)
    # We want to minimize P(S) while maximizing Sum(1/x for x in S).
    # Actually, the constraint is "without any nonconstant 3-AP".
    # So P(S) must be 0.
    
    # Local Search:
    # Start with a valid set (e.g., greedy).
    # Move: Add x, Remove x, Swap x with y.
    # Objective: Maximize Sum(1/x).
    # Constraint: No 3-APs.
    
    # Since we must maintain validity, we can use a penalty method if we allow temporary invalidity,
    # or strictly valid moves.
    # Strictly valid moves:
    # - Add x: Check if adding x creates any 3-AP. 
    #   This requires checking all pairs (y, z) in S such that {x, y, z} is an AP.
    #   Precompute for each x, the list of (y, z) pairs.
    
    # Let's precompute:
    # For each x, list of (y, z) such that {x, y, z} is a 3-AP.
    # If x is the middle term: y < x < z, 2x = y+z.
    # If x is the first term: x < y < z, z = 2y - x.
    # If x is the last term: y < z < x, y = 2z - x.
    
    # Actually, simpler:
    # For each x, store a list of tuples (y, z) where y < z and {x, y, z} is a 3-AP.
    # When adding x, we need to ensure that for no (y, z) in this list are both y and z in S.
    
    # Precomputation:
    # conflict_pairs = {x: [(y1, z1), (y2, z2), ...]}
    
    conflict_pairs = {i: [] for i in range(1, n+1)}
    for a, b, c in ap_list:
        # a, b, c form an AP
        # a is involved with (b, c)
        conflict_pairs[a].append((b, c))
        # b is involved with (a, c)
        conflict_pairs[b].append((a, c))
        # c is involved with (a, b)
        conflict_pairs[c].append((a, b))
        
    # Note: (b, c) means if b and c are in S, a cannot be added.
    
    # Greedy initialization
    S = set()
    for x in range(1, n+1):
        # Check if x can be added
        can_add = True
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                can_add = False
                break
        if can_add:
            S.add(x)
            
    current_sum = sum(1.0/x for x in S)
    best_sum = current_sum
    best_set = S.copy()
    
    # Local Search / Hill Climbing with Swaps
    # We try to improve the sum by swapping elements.
    # A swap (remove u, add v) is valid if:
    # 1. u is in S, v is not in S.
    # 2. S \ {u} is valid (always true if S was valid? No, removing an element never creates an AP).
    # 3. (S \ {u}) ∪ {v} is valid.
    
    # To check validity of (S \ {u}) ∪ {v}:
    # We need to check if v creates an AP with any pair in (S \ {u}).
    # This is similar to checking if v can be added to S \ {u}.
    
    # Optimization:
    # Precompute for each v, the list of pairs.
    # When checking v against S' = S \ {u}:
    # For each (y, z) in conflict_pairs[v]:
    #   If y in S' and z in S', then invalid.
    
    # Since S' is almost S, we can check:
    # If y in S and z in S, and not (y==u or z==u), then invalid.
    
    # Let's implement a helper to check if v can be added to a set T.
    # But T changes.
    
    # Let's try a simple hill climbing with random restarts or simulated annealing.
    # Given time limit, simple hill climbing with random swaps might be enough.
    
    # We will perform a series of swap attempts.
    # For each attempt:
    # Pick a random u in S and random v not in S.
    # If 1/v > 1/u (i.e., v < u), it's a potential improvement in sum.
    # Check validity.
    
    # To make it faster, we can prioritize swaps where v < u.
    # We can iterate through candidates.
    
    # Let's use a simple random search with acceptance of improving moves.
    
    # To speed up validity check, we can maintain a count of "conflicts" or just check directly.
    # Direct check is O(deg(v)). Max degree is small.
    
    # Let's run for a fixed number of iterations or until time limit.
    
    # We'll use a simple loop.
    
    # To avoid getting stuck in local minima, we can occasionally accept worse moves (Simulated Annealing).
    # But since we need to maximize sum, and constraints are hard, let's stick to improving moves first.
    
    # Actually, the greedy set is likely quite good. Swaps that replace a large number with a small number
    # (which has larger reciprocal) are the main source of improvement.
    
    # Let's try to systematically try all pairs (u, v) where v < u?
    # That is O(|S| * (n-|S|)) which is ~ 100 * 150 = 15000 checks.
    # Each check is O(deg(v)) ~ 10-20.
    # So 15000 * 20 = 300,000 ops per pass.
    # We can do many passes.
    
    # Let's do passes until no improvement or time limit.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to avoid bias? Or just iterate.
        # Let's iterate u in S and v in [1..n]\S.
        # To find best swap, we might need to check all.
        
        # Optimization: Only consider v < u.
        # If v > u, 1/v < 1/u, so sum decreases. We only want to increase sum.
        # So we only look for v < u.
        
        for u in list(S):
            # Find best v < u not in S
            best_v = -1
            best_gain = 0
            # We want to maximize 1/v - 1/u.
            # This is maximized when v is as small as possible.
            # So we should check v from 1 upwards?
            # But v must not be in S and must be valid.
            
            # Let's check v from 1 to u-1.
            # If we find a valid v, is it the best?
            # Yes, because 1/v is decreasing. Smallest v gives largest 1/v.
            # So for a fixed u, the best v is the smallest valid v < u not in S.
            
            for v in range(1, u):
                if v in S:
                    continue
                # Check if v can be added to S \ {u}
                # Check conflicts of v
                valid = True
                for y, z in conflict_pairs[v]:
                    # y, z are in S \ {u} ?
                    # If y == u or z == u, it's removed.
                    # So we need y in S and z in S, and y != u and z != u.
                    if y in S and z in S:
                        if y != u and z != u:
                            valid = False
                            break
                if valid:
                    # This v is valid.
                    # Since we iterate v from 1, this is the smallest valid v.
                    # So this is the best swap for this u.
                    # Gain: 1/v - 1/u
                    # We accept this swap.
                    S.remove(u)
                    S.add(v)
                    improved = True
                    current_sum += 1.0/v - 1.0/u
                    break # Move to next u
            
            if time.time() > deadline:
                break
                
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = S.copy()
            
    return list(best_set)

# EVOLVE-BLOCK-END
