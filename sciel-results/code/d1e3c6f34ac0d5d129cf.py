"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a, b) > 256 for all distinct pairs a, b.
    Maximize the exact reciprocal sum (sum of 1/x for x in the subset).
    """
    from math import lcm
    from fractions import Fraction
    import sys
    from functools import lru_cache

    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute conflict graph
    # conflict[i][j] is True if lcm(candidates[i], candidates[j]) <= 256
    # We want an independent set in this conflict graph.
    # This is equivalent to a clique in the complement graph.
    
    # Let's use the complement graph approach with Bron-Kerbosch.
    # But n=255 is too large for naive Bron-Kerbosch without strong bounds.
    
    # Alternative: Weighted Independent Set via Branch and Bound.
    # Since weights are 1/x, smaller x have larger weight.
    
    # Let's try a different approach: 
    # The condition lcm(a,b) > 256 is equivalent to: 
    # For any distinct a, b in the set, they do not divide a common number <= 256? No.
    # lcm(a,b) = a*b / gcd(a,b).
    # lcm(a,b) <= 256 means a*b <= 256 * gcd(a,b).
    
    # Let's stick to the complement graph clique finding.
    # We need max weight clique in the graph where edge exists if lcm(a,b) > 256.
    
    # Build adjacency list for the "good" graph (complement of conflict)
    # adj[i] = set of j such that i != j and lcm(candidates[i], candidates[j]) > 256
    
    # To make it faster, we can use bitmasks or sets.
    # n=255 is large. Bitmasks won't fit in a single integer easily (255 bits), but Python supports big ints.
    
    # Let's use sets of indices for the Bron-Kerbosch algorithm.
    
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(candidates[i], candidates[j]) > 256:
                adj[i].add(j)
                adj[j].add(i)
                
    weights = [Fraction(1, x) for x in candidates]
    
    best_weight = Fraction(0)
    best_clique = []
    
    # Bron-Kerbosch with pivot and coloring bound
    # To keep it simple and within time limits, we might need to be careful.
    # However, 255 nodes is quite large for exact clique.
    
    # Maybe we can prune more aggressively.
    # Note that if a number is prime > 128, its lcm with any other number > 1 is > 256?
    # If p > 128 and q >= 2, lcm(p,q) >= p > 128. Is it always > 256?
    # If p=131, q=2, lcm=262 > 256.
    # If p=131, q=131 (not distinct).
    # If p=129 (not prime), q=2, lcm=258 > 256.
    # If p=128, q=2, lcm=128 <= 256.
    # So numbers > 128 are compatible with almost everything?
    # Let's check: if a > 128 and b > 1, lcm(a,b) >= a > 128.
    # Can lcm(a,b) be <= 256?
    # Only if b is a divisor of a * k where k is small?
    # Actually, if a > 128, and b divides a, then lcm(a,b) = a > 128.
    # If b does not divide a, lcm(a,b) is a multiple of a, so >= 2a > 256.
    # Wait, if b divides a, lcm is a. Since a > 128, a <= 256 is possible.
    # Example: a=256, b=128. lcm(256, 128) = 256. NOT > 256.
    # So a=256 conflicts with 128, 64, 32, ... divisors.
    
    # This suggests that large numbers are "safe" except with their divisors.
    
    # Let's try a greedy heuristic first to get a good lower bound, then branch and bound.
    
    # Greedy: Sort by weight desc (x asc). Add if compatible.
    selected = []
    selected_set = set()
    
    for x in candidates:
        compatible = True
        for y in selected:
            if lcm(x, y) <= 256:
                compatible = False
                break
        if compatible:
            selected.append(x)
            selected_set.add(x)
            
    current_best_weight = sum(Fraction(1, x) for x in selected)
    best_clique = selected[:]
    
    # Now, can we improve?
    # The greedy set includes 2, 3, 5, 7, 11, ... primes?
    # 2 conflicts with 4, 6, 8, ...
    # 3 conflicts with 6, 9, ...
    # Actually, if we pick 2, we can't pick any even number.
    # If we pick 3, we can't pick multiples of 3.
    
    # Let's try to solve it exactly using a recursive search with pruning.
    # Given the time limit of 60s, and n=255, exact solution might be tough.
    # But the structure is specific.
    
    # Let's try a different exact approach:
    # Since we want to maximize sum(1/x), we prefer small x.
    # However, small x conflict with many others.
    
    # Let's try to find the optimal set by considering the constraints.
    # If we include 2, we exclude all even numbers.
    # If we exclude 2, we can include even numbers, but 4, 6, etc.
    
    # Let's try to branch on whether to include 2 or not.
    # This is getting complex.
    
    # Let's stick to the Bron-Kerbosch but with a good initial bound.
    # Or, simpler: just return the greedy result if it's good enough?
    # The task asks to maximize.
    
    # Let's try a local search improvement on the greedy set.
    # Swap: Remove a set of conflicting nodes, add a new one.
    
    improved = True
    while improved:
        improved = False
        for x in candidates:
            if x in selected_set:
                continue
            conflicts = [y for y in selected if lcm(x, y) <= 256]
            if not conflicts:
                selected.append(x)
                selected_set.add(x)
                improved = True
            else:
                loss = sum(Fraction(1, y) for y in conflicts)
                gain = Fraction(1, x)
                if gain > loss:
                    for y in conflicts:
                        selected.remove(y)
                        selected_set.remove(y)
                    selected.append(x)
                    selected_set.add(x)
                    improved = True
                    
    return selected
# EVOLVE-BLOCK-END
