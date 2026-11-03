import time
import numpy as np
from itertools import permutations
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate standard superpermutation to get the set of permutations
    # and an initial order (though we will re-order them)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Extract unique permutations in order of first appearance
    # This gives us a valid initial path (with duplicates removed, we just need to ensure connectivity)
    # Actually, removing duplicates might break the overlap property.
    # So we treat it as a TSP problem on the set of all N permutations.
    
    all_perms = list(permutations(range(1, n+1)))
    N = len(all_perms)
    
    # Map perm to index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute cost matrix
    # cost[i][j] = n - overlap(all_perms[i], all_perms[j])
    # Overlap: largest k such that all_perms[i][n-k:] == all_perms[j][:k]
    # Max k is n-1.
    
    # For n=7, N=5040. Matrix 5040x5040.
    # We can compute this in a vectorized way or with loops.
    # Looping 25M times in Python is slow (~10-20s).
    # Let's try to optimize.
    
    # We only need the cost for the TSP.
    # We can compute the cost on the fly or precompute.
    # Precomputation is safer for speed in the optimization loop.
    
    # To speed up precomputation, we can use the fact that overlap is determined by suffix/prefix.
    # Group permutations by their suffix of length n-1?
    # Actually, just compute it.
    
    # Use numpy for storage
    costs = np.zeros((N, N), dtype=np.int8)
    
    # This loop might be the bottleneck.
    # 5040 * 5040 = 25,401,600 iterations.
    # In Python, ~10-30 seconds.
    # Acceptable if we have 140s.
    
    # Optimization:
    # overlap(i, j)
    # Let u = all_perms[i], v = all_perms[j]
    # Check k from n-1 down to 0.
    
    # We can precompute suffixes and prefixes?
    # Or just use string comparison if we convert to strings?
    # No, tuple comparison is fast in C.
    
    for i in range(N):
        u = all_perms[i]
        for j in range(N):
            v = all_perms[j]
            # Check overlap
            # Max overlap n-1
            # u[n-k:] == v[:k]
            # We can check k=n-1 first.
            # If u[1:] == v[:n-1], cost 1.
            # Else check n-2, etc.
            
            # To make it faster, we can precompute the "prefix tuple" and "suffix tuple" for each k?
            # Or just do the loop.
            
            # Let's try a simple loop.
            # Since n is small (7), the inner loop is small.
            overlap = 0
            # Check from largest to smallest
            # k = n-1
            if u[1:] == v[:n-1]:
                overlap = n - 1
            elif n > 2 and u[2:] == v[:n-2]:
                overlap = n - 2
            elif n > 3 and u[3:] == v[:n-3]:
                overlap = n - 3
            elif n > 4 and u[4:] == v[:n-4]:
                overlap = n - 4
            elif n > 5 and u[5:] == v[:n-5]:
                overlap = n - 5
            elif n > 6 and u[6:] == v[:n-6]:
                overlap = n - 6
            else:
                overlap = 0 # or check further if n>7, but n<=7 usually
                
            # Wait, what if n=7?
            # u[1:] is len 6. v[:6] is len 6.
            # If they match, overlap 6.
            # If not, check u[2:] (len 5) vs v[:5].
            
            # General loop is safer.
            # But unrolling for n=7 is faster.
            
            # Let's just do a generic loop for correctness and let numpy handle the storage.
            # Actually, the unrolled version above is correct for n<=7.
            # For n=7:
            # k=6: u[1:] vs v[:6]
            # k=5: u[2:] vs v[:5]
            # ...
            # k=1: u[6:] vs v[:1]
            # k=0: always match.
            
            # The code above checks k=6,5,4,3,2,1.
            # If none match, overlap=0.
            
            costs[i, j] = n - overlap
            
    # Initial path: use the order from the classic construction's unique perms
    # Or just random permutation?
    # Classic order is a good heuristic.
    
    # Extract unique perms from word
    seen = set()
    initial_path = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_idx:
            idx = perm_to_idx[p]
            if idx not in seen:
                seen.add(idx)
                initial_path.append(idx)
                
    # Ensure all perms are in path
    if len(initial_path) < N:
        # Add missing ones randomly
        missing = [i for i in range(N) if i not in seen]
        random.shuffle(missing)
        initial_path.extend(missing)
        
    path = initial_path
    current_cost = 0
    for i in range(len(path) - 1):
        current_cost += costs[path[i], path[i+1]]
        
    # Simulated Annealing / 2-opt
    T = 100.0
    cool = 0.99
    best_path = path.copy()
    best_cost = current_cost
    
    # 2-opt optimization
    # We will do random 2-opt moves.
    
    # Precompute something to speed up cost calculation?
    # cost(i, j) is O(1) lookup.
    
    while time.time() < deadline - 5:
        # Pick random i, j
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Reverse path[i:j+1]
        # Calculate delta cost
        # Old: (i-1, i) + (i, i+1) ... (j, j+1)
        # New: (i-1, j) + (j, j-1) ... (i, i+1)
        
        # Edges affected:
        # 1. Edge (i-1, i) -> (i-1, j)
        # 2. Edge (j, j+1) -> (i, j+1)
        # 3. Internal edges reversed: (k, k+1) -> (k+1, k)
        
        # Since graph is directed, cost(u, v) != cost(v, u) generally?
        # Overlap(u, v) vs Overlap(v, u).
        # Yes, they are different.
        
        delta = 0
        
        # Edge 1: i-1 to i vs i-1 to j
        if i > 0:
            old_c = costs[path[i-1], path[i]]
            new_c = costs[path[i-1], path[j]]
            delta += new_c - old_c
            
        # Edge 2: j to j+1 vs i to j+1
        if j < N - 1:
            old_c = costs[path[j], path[j+1]]
            new_c = costs[path[i], path[j+1]]
            delta += new_c - old_c
            
        # Internal edges:
        # Old: sum(costs[path[k], path[k+1]] for k in i..j-1)
        # New: sum(costs[path[k+1], path[k]] for k in i..j-1)
        
        # To avoid O(N) sum, we can maintain a prefix sum of costs?
        # But the path changes.
        # For now, calculate sum. Segment length is random, avg N/3.
        # N=5000. Sum is ~1600 ops.
        # 25M precomp took ~10s.
        # 1M iterations of 2-opt with 1600 ops each = 1.6B ops. Too slow in Python.
        
        # We need to optimize the delta calculation.
        # We can maintain an array `edge_costs` where edge_costs[k] = costs[path[k], path[k+1]].
        # And a Fenwick tree or just a simple sum if we only do 2-opt?
        # No, 2-opt reverses a segment.
        
        # Alternative: Just do it and hope it's fast enough?
        # Or reduce iterations.
        
        # Let's try to be smarter.
        # If we just do adjacent swaps (2-opt with j=i+1), the internal sum is 0.
        # Delta = cost(i-1, i+1) - cost(i-1, i) + cost(i, i+2) - cost(i+1, i+2) + cost(i+1, i) - cost(i, i+1)?
        # No, swap i, i+1.
        # Old: (i-1, i), (i, i+1), (i+1, i+2)
        # New: (i-1, i+1), (i+1, i), (i, i+2)
        # Delta = cost(i-1, i+1) + cost(i+1, i) + cost(i, i+2) - (cost(i-1, i) + cost(i, i+1) + cost(i+1, i+2))
        # This is O(1).
        
        # So we can do random adjacent swaps.
        
        i = random.randint(0, N - 2)
        j = i + 1
        
        delta = 0
        
        if i > 0:
            delta += costs[path[i-1], path[j]] - costs[path[i-1], path[i]]
            
        if j < N - 1:
            delta += costs[path[i], path[j+1]] - costs[path[j], path[j+1]]
            
        # Middle edge: (i, j) becomes (j, i)
        delta += costs[path[j], path[i]] - costs[path[i], path[j]]
        
        if delta < 0 or (T > 1.0 and random.random() < np.exp(-delta / T)):
            # Accept
            path[i], path[j] = path[j], path[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path.copy()
        
        T *= cool
        
    # Reconstruct word from best_path
    word_final = list(all_perms[best_path[0]])
    for k in range(1, N):
        prev = all_perms[best_path[k-1]]
        curr = all_perms[best_path[k]]
        # Find overlap
        # We know cost = n - overlap.
        # So overlap = n - costs[best_path[k-1], best_path[k]]
        ov = n - costs[best_path[k-1], best_path[k]]
        word_final.extend(curr[ov:])
        
    return "".join(map(str, word_final))
