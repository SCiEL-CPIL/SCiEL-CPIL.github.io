import numpy as np
from itertools import combinations
import time

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total_subsets = 1 << N  # 4096
    
    # Precompute comparability edges: edge (a,b) exists if a is proper subset of b
    # Only store edges where a < b (canonical ordering)
    # For each subset a, find all proper supersets b > a
    edges = []
    edge_indices = {}  # (a,b) -> edge index
    
    # Precompute for each subset its mask value
    for a in range(total_subsets):
        # Find all proper supersets of a
        # A superset b of a satisfies: (a & b) == a and b != a
        # Iterate over all b > a that are supersets
        remaining = ((1 << N) - 1) ^ a
        # Generate all non-empty subsets of remaining
        sub = remaining
        while sub:
            b = a | sub
            if b > a:
                edge_indices[(a, b)] = len(edges)
                edges.append((a, b))
            sub = (sub - 1) & remaining
        # Also b can equal a (skip, we want proper)
        # Actually we need b to be a PROPER superset, so b != a is required
        # But sub can be 0 which gives b=a, we skip that
    
    # Wait, the above doesn't generate b=a case since sub starts at remaining
    # and we skip sub=0. Good.
    # But we need to make sure we only count each pair once.
    # Since we only generate b > a, each edge appears once.
    
    E = len(edges)
    
    # Build adjacency: for each subset, list of edges incident to it
    # And for each edge, the two endpoint subsets
    adj = [[] for _ in range(total_subsets)]
    for ei, (a, b) in enumerate(edges):
        adj[a].append(ei)
        adj[b].append(ei)
    
    # ILP formulation:
    # Variables: x[s] for s in range(total_subsets), binary
    # Variables: y[e] for e in range(E), binary
    # Maximize: sum(y[e])
    # Subject to:
    #   y[e] <= x[a] for each edge e=(a,b)
    #   y[e] <= x[b] for each edge e=(a,b)
    #   sum(x[s]) = M
    #   x[s] in {0,1}, y[e] in {0,1}
    
    # Number of binary variables: total_subsets + E
    # This is large. total_subsets = 4096, E = ~33M
    # This is too large for a dense ILP.
    
    # Alternative: Use a different approach.
    # Since we need exactly M=40 subsets out of 4096, and edges are dense,
    # let's use a heuristic + local search approach.
    
    # Heuristic: Start with colex initial segment, then hill climb.
    
    # Colex initial segment of size M from 2^N
    # In colex order, the first M subsets are:
    # The colex ordering lists subsets by their largest element first, then recursively.
    # The first M subsets in colex are simply: for i in range(M), the subset with
    # binary representation of i (in colex sense).
    
    # Actually, let's use a different initial solution: 
    # Take the M subsets with smallest "colex rank" or just use the first M
    # subsets in colex order.
    
    # Colex order: subset S has colex rank = sum over elements e in S of 2^e
    # Wait, that's just the binary value. In colex, subsets are ordered by:
    # 1. Size (or not?)
    # Actually, colex order: compare the largest element where they differ.
    # The first M subsets in colex order are {0,1,...,k} for appropriate k,
    # then more subsets.
    
    # Simpler: just use the first M subsets in binary order (0 to M-1)
    # as initial solution, then hill climb.
    
    # Actually, let's use a better initial: the colex initial segment.
    # The colex initial segment of size M consists of the M smallest subsets
    # in colex order. In colex, the order is:
    # {}, {0}, {1}, {0,1}, {2}, {0,2}, {1,2}, {0,1,2}, {3}, ...
    
    # Let me just use binary order 0..M-1 as initial, then improve.
    
    def score_solution(selected):
        """Compute number of comparable pairs in selected subsets."""
        sel_set = set(selected)
        count = 0
        for a in selected:
            for b in selected:
                if a < b and (a & b) == a:
                    count += 1
        return count
    
    def hill_climb(selected, max_iter=50000):
        """Local search: try swapping one element in with one out."""
        sel_set = set(selected)
        current_score = score_solution(selected)
        
        for iteration in range(max_iter):
            if time.time() - t0 > 50:
                break
            
            best_gain = 0
            best_swap = None
            
            # Try all possible swaps: remove i from selected, add j not in selected
            # Sample a subset of swaps to keep it fast
            # Try removing each selected element and adding a few random non-selected
            
            # First, compute the contribution of each selected element
            # When we remove element a, we lose all edges (a,b) where b is in selected
            # and a < b, plus edges (b,a) where b < a and b is in selected
            
            # Let's precompute for each selected element its "degree" within selected
            deg_in = [0] * len(selected)
            # Actually, let's just do random swaps for speed
            
            # Try up to 200 random swaps per iteration
            for _ in range(200):
                i = np.random.randint(len(selected))
                # Pick a random non-selected subset
                j = np.random.randint(total_subsets)
                if j in sel_set:
                    continue
                
                a = selected[i]
                # Compute score change: remove a, add j
                # Lost: all edges involving a within selected
                # Gained: all edges involving j within (selected - {a} + {j})
                
                # Compute edges lost by removing a
                lost = 0
                for b in selected:
                    if b == a:
                        continue
                    if b > a and (a & b) == a:
                        lost += 1
                    elif b < a and (b & a) == b:
                        lost += 1
                
                # Compute edges gained by adding j
                gained = 0
                for b in selected:
                    if b == a:
                        continue
                    if b > j and (j & b) == j:
                        gained += 1
                    elif b < j and (b & j) == b:
                        gained += 1
                
                delta = gained - lost
                if delta > best_gain:
                    best_gain = delta
                    best_swap = (i, j)
            
            if best_swap is not None and best_gain > 0:
                i, j = best_swap
                a = selected[i]
                selected[i] = j
                sel_set.remove(a)
                sel_set.add(j)
                current_score += best_gain
            else:
                # No improvement found in this iteration
                # Try a random restart-like perturbation
                break
        
        return selected, current_score
    
    # Initial solution: first M subsets in binary order
    selected = list(range(M))
    selected, current_score = hill_climb(selected, max_iter=1000)
    
    # Try multiple random restarts
    best_selected = selected[:]
    best_score = current_score
    
    while time.time() - t0 < 52:
        # Random restart
        selected = np.random.permutation(total_subsets)[:M].tolist()
        selected, current_score = hill_climb(selected, max_iter=5000)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected[:]
    
    # Final polish
    best_selected, best_score = hill_climb(best_selected, max_iter=20000)
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
