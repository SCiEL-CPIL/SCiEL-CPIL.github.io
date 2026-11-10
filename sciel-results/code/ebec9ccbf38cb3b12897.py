import math
import time

def construct():
    """
    Find a large pairwise-noncoprime subset of [1, 255] containing 255.
    
    Strategy:
    1. 255 = 3 * 5 * 17. Any element in the set must share a factor with 255 (otherwise gcd with 255 is 1).
    2. Thus, the set must be a subset of numbers in [1, 255] that are divisible by 3, 5, or 17.
    3. We want to maximize the size of a clique in the graph where vertices are these candidates and edges exist if gcd(u, v) > 1.
    4. A simple baseline is taking all multiples of 3 (size 85).
    5. We can potentially do better by mixing multiples of 3, 5, and 17, but we must ensure pairwise gcd > 1.
       - Multiples of 3 and 5: gcd > 1 only if they share a factor. Since 3 and 5 are coprime, a multiple of 3 and a multiple of 5 share a factor > 1 iff the multiple of 3 is also a multiple of 5 (i.e., multiple of 15) OR the multiple of 5 is also a multiple of 3. Wait, no.
       Let A be the set of multiples of 3. Let B be the set of multiples of 5.
       If x in A and y in B, gcd(x, y) > 1.
       If x is a multiple of 3 but not 5, and y is a multiple of 5 but not 3, then gcd(x, y) could be 1 (e.g., x=3, y=5, gcd=1).
       So we cannot just take all multiples of 3 and all multiples of 5.
       
    6. However, we can take all multiples of 3, and add some multiples of 5 that are also multiples of 3 (i.e., multiples of 15) -- but those are already in the set of multiples of 3.
       Or add multiples of 5 that share a factor with ALL multiples of 3 in the set.
       This is a clique problem.
       
    7. Let's use a greedy heuristic to expand from the set of multiples of 3.
       Start with S = {multiples of 3 in [1, 255]}. Size 85.
       Try to add other candidates (multiples of 5 or 17 not in S) one by one.
       A candidate c can be added if gcd(c, s) > 1 for all s in S.
       
    8. Actually, a smarter approach:
       The set must be a clique.
       Consider the prime factors 3, 5, 17.
       If we pick a number, it must be divisible by at least one of these.
       If we pick two numbers, they must share a prime factor.
       
       Let's try to find the maximum clique in the graph of candidates (divisible by 3, 5, or 17).
       Number of candidates:
       Multiples of 3: 85
       Multiples of 5: 51
       Multiples of 17: 15
       Union: 85 + 51 + 15 - (multiples of 15) - (multiples of 51) - (multiples of 85) + (multiples of 255)
       Multiples of 15: 17
       Multiples of 51: 5
       Multiples of 85: 3
       Multiples of 255: 1
       Union size = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127.
       
       We need to find a clique of max size in this graph of 127 nodes.
       A clique of size 85 exists (multiples of 3).
       Can we get larger?
       
       Let's try a greedy maximum clique algorithm.
       1. Order candidates by degree (number of neighbors) descending.
       2. Greedily add nodes that connect to all previously added nodes.
       
       Or use a branch and bound / Bron-Kerbosch with pivoting.
       Given the small size (127 nodes), a simple greedy or even a limited backtracking might work within 60s.
       
       Let's implement a greedy approach first, then try to improve.
       
    9. Greedy:
       - Start with empty set S.
       - Sort candidates by degree descending.
       - For each candidate c in sorted order:
           - If c is compatible with all elements in S, add c to S.
       
       This might not find the maximum clique, but it's a good heuristic.
       
       To improve, we can try multiple orderings or use a local search.
       
       Let's also consider that the set of multiples of 3 is a very strong clique.
       Maybe we can swap some multiples of 3 with other numbers to increase size? Unlikely, as 85 is quite large.
       
       Let's check if we can add any non-multiple-of-3 to the set of multiples of 3.
       Let S = {multiples of 3}.
       Can we add a multiple of 5, say 5?
       gcd(5, 3) = 1. No.
       Can we add a multiple of 17, say 17?
       gcd(17, 3) = 1. No.
       
       So we cannot add any number that is not a multiple of 3 to the set of ALL multiples of 3.
       But maybe if we remove some multiples of 3, we can add more numbers?
       
       For example, remove 3. Can we add 5?
       S = {multiples of 3 except 3} U {5}.
       Check gcd(5, s) for all s in S.
       gcd(5, 6) = 1. No.
       
       It seems hard to beat 85.
       
       However, let's try to find a clique that is not just multiples of a single prime.
       For instance, multiples of 15 (size 17) plus...
       
       Let's run the greedy algorithm on the 127 candidates.
    """
    start_time = time.time()
    deadline = 90  # seconds
    
    # 1. Generate candidates
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    n = len(candidates)
    
    # 2. Precompute adjacency
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Greedy Maximum Clique
    # Sort by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    clique_indices = []
    for idx in order:
        # Check if idx is compatible with all in clique_indices
        compatible = True
        for c_idx in clique_indices:
            if c_idx not in adj[idx]:
                compatible = False
                break
        if compatible:
            clique_indices.append(idx)
            
    # Convert back to values
    clique = [candidates[i] for i in clique_indices]
    
    # 4. Try to improve with local search / swapping?
    # Or try multiple random orderings?
    
    # Let's try a few random orderings to see if we get a better clique
    import random
    best_clique_size = len(clique)
    best_clique = clique
    
    # Deterministic seed for reproducibility
    random.seed(42)
    
    for _ in range(50):  # Try 50 random orderings
        if time.time() > deadline - 5:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        
        c_clique = []
        for idx in random_order:
            compatible = True
            for c_idx in c_clique:
                if c_idx not in adj[idx]:
                    compatible = False
                    break
            if compatible:
                c_clique.append(idx)
        
        if len(c_clique) > best_clique_size:
            best_clique_size = len(c_clique)
            best_clique = [candidates[i] for i in c_clique]
            
    # Ensure 255 is in the set
    if 255 not in best_clique:
        # If 255 is not in the clique, we have a problem.
        # But 255 is divisible by 3, 5, 17.
        # It should have high degree.
        # If it's not in the greedy clique, we might need to force it.
        # But the greedy algorithm should pick high-degree nodes first.
        # Let's check if we can add 255 to the current clique.
        idx_255 = candidates.index(255)
        compatible = True
        for c_idx in [candidates.index(v) for v in best_clique]:
            if c_idx not in adj[idx_255]:
                compatible = False
                break
        if compatible:
            best_clique.append(255)
        else:
            # Fallback: return multiples of 3
            best_clique = [i for i in range(1, 256) if i % 3 == 0]
            
    return best_clique
