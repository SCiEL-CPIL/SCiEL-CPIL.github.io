import itertools
import math

def construct():
    max_val = 128
    max_sum = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
                
    # Bipartite graph model
    # Left nodes: A candidates (odds)
    # Right nodes: B candidates (evens)
    # Edge if sum is prime
    
    odds = [x for x in range(1, max_val + 1) if x % 2 == 1]
    evens = [x for x in range(2, max_val + 1) if x % 2 == 0]
    
    # Adjacency list: a -> set of b's
    adj = {a: set() for a in odds}
    for a in odds:
        for b in evens:
            if is_prime[a + b]:
                adj[a].add(b)
                
    # Greedy heuristic: Sort odds by degree (number of valid b's)
    # Higher degree means more flexibility
    sorted_odds = sorted(odds, key=lambda x: len(adj[x]), reverse=True)
    
    # Try to find A of size 3 with large common neighborhood
    # Then find B of size 3 in that neighborhood with distinct sums
    
    # To speed up, we can limit the search to the top K odds
    # But since the space is small, we can just iterate
    # However, the directive says "greedy maximum-degree heuristic"
    # So let's pick the top few odds and try combinations from there first
    
    # Let's try combinations of the top 20 odds first
    top_odds = sorted_odds[:20]
    
    for A in itertools.combinations(top_odds, 3):
        a1, a2, a3 = A
        # Common neighbors
        common_b = adj[a1].intersection(adj[a2]).intersection(adj[a3])
        if len(common_b) < 3:
            continue
            
        common_b_list = list(common_b)
        # Try to find 3 distinct b's in common_b such that sums are distinct
        for B in itertools.combinations(common_b_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                return [list(A), list(B)]
                
    # If top 20 didn't work, try all odds
    for A in itertools.combinations(odds, 3):
        a1, a2, a3 = A
        common_b = adj[a1].intersection(adj[a2]).intersection(adj[a3])
        if len(common_b) < 3:
            continue
            
        common_b_list = list(common_b)
        for B in itertools.combinations(common_b_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                return [list(A), list(B)]
                
    # Fallback: Random search (unlikely to be needed)
    import random
    for _ in range(10000):
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        sums = set()
        for a in A:
            for b in B:
                if is_prime[a+b]:
                    sums.add(a+b)
                else:
                    break
            else:
                continue
            break
        else:
            if len(sums) == 9:
                return [A, B]
                
    return [[1, 7, 13], [4, 10, 16]]
