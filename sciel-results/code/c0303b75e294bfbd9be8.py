"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt, log

def construct():
    deadline = time.time() + 80.0  # 80s budget, 10s safety margin
    N = 512
    
    # Precompute factorizations for all numbers 1..N
    # Factorize each number into prime powers
    # For lcm(a,b) <= N, we need for each prime p: max(v_p(a), v_p(b)) such that product <= N
    # This is complex. Instead, use a greedy/set-cover approach.
    
    # Key insight: lcm(a,b) = a*b / gcd(a,b)
    # So lcm(a,b) <= N iff a*b <= N * gcd(a,b)
    # Equivalently: for all a,b in S, a*b/gcd(a,b) <= N
    
    # Strategy: 
    # 1. Start with all numbers 1..N
    # 2. Repeatedly remove the element that is in the most "conflicting" pairs
    # 3. Or: build up greedily, adding elements that don't conflict
    
    # Better: use the fact that if we group by "core" (odd part), 
    # numbers of the form odd * 2^k have lcm = odd * 2^max(k1,k2) * ... 
    # Actually lcm(odd*2^a, odd*2^b) = odd * 2^max(a,b) if same odd part
    # But different odd parts complicate things.
    
    # Let's use a different approach: 
    # For each number x in 1..N, compute which other numbers y have lcm(x,y) > N
    # Then find a maximum independent set in the conflict graph (or minimum vertex cover)
    
    # Conflict graph: edge between x,y if lcm(x,y) > N
    # We want maximum independent set. This is NP-hard in general, but N=512 is small enough
    # for heuristic approaches.
    
    # Let's build the conflict set for each number
    # conflict[i] = set of j where lcm(i,j) > N (using 1-indexed, i from 1 to N)
    
    # Precompute lcm check efficiently
    # lcm(a,b) = a // gcd(a,b) * b
    # Check if a // gcd(a,b) * b <= N, i.e., a * b <= N * gcd(a,b)
    
    # For speed, precompute gcd table or use math.gcd
    # N=512, so pairs are ~131k, which is fine
    
    # Build conflict sets
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            # lcm = a // g * b
            if a // g * b > N:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Now find maximum independent set in this graph
    # Heuristic: greedy coloring / local search
    
    # Start with all nodes, then remove one endpoint from each edge
    # Greedy: repeatedly pick the node with highest degree, remove it and its neighbors? No, that's vertex cover.
    # For independent set: pick a node, keep it, remove all its neighbors from consideration.
    
    # Actually, let's try: start with empty set, greedily add nodes in order of lowest degree first
    # This tends to give larger independent sets
    
    # Compute degrees
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    
    # Greedy: sort by degree ascending, add if no conflict with already added
    order = sorted(range(1, N + 1), key=lambda x: degrees[x])
    
    selected = set()
    for node in order:
        # Check if node conflicts with any selected node
        conflict = False
        for sel in selected:
            if node in conflicts[sel]:
                conflict = True
                break
        if not conflict:
            selected.add(node)
    
    current_best = len(selected)
    best_set = set(selected)
    
    # Local search: try to improve by swapping
    # For each node not in set, try to add it and remove conflicting nodes if net gain > 0
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding each unselected node
        for node in range(1, N + 1):
            if node in best_set:
                continue
            # Find conflicts with current set
            conflicting = [s for s in best_set if node in conflicts[s]]
            if len(conflicting) == 0:
                # Can add directly
                best_set.add(node)
                improved = True
            elif len(conflicting) == 1:
                # Try swapping: remove one conflicting, add node
                # Only beneficial if we can later add more, but for now just try
                pass
            # For simplicity, only do direct additions and 1-for-1 swaps that increase size
            # Actually, let's do a more thorough local search
    
    # Better local search: for each pair (in_node, out_node), try swapping
    # But that's O(N^2) per pass which is fine for N=512
    
    # Let's do a proper local search
    best_set = set(best_set)
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        progress = False
        # Try to add any node that has no conflicts
        for node in range(1, N + 1):
            if node in best_set:
                continue
            if all(node not in conflicts[s] for s in best_set):
                best_set.add(node)
                progress = True
        
        # Try 1-for-1 swaps: remove one node, add another that was previously blocked
        # For each node not in set, find which nodes in set block it
        for node in range(1, N + 1):
            if node in best_set:
                continue
            blockers = [s for s in best_set if node in conflicts[s]]
            if len(blockers) == 1:
                blocker = blockers[0]
                # If we remove blocker, can we add node and maybe others?
                # Simple: just swap if it doesn't reduce size (it keeps same size)
                # But we need the new set to be valid. Removing blocker and adding node:
                # node must not conflict with any other node in best_set \ {blocker}
                # And blocker's removal must not cause issues (it won't, since we're removing)
                if all(node not in conflicts[s] for s in best_set if s != blocker):
                    # Check if this swap is beneficial: it keeps size same, but might enable future additions
                    # For now, let's not do neutral swaps to avoid cycles
                    pass
        
        # Try 2-for-1 swaps: remove two blockers, add one node
        for node in range(1, N + 1):
            if node in best_set:
                continue
            blockers = [s for s in best_set if node in conflicts[s]]
            if len(blockers) == 2:
                b1, b2 = blockers
                # Remove b1 and b2, add node: net -1, bad
                pass
            if len(blockers) == 3:
                # Remove 3, add 1: net -2, bad
                pass
        
        if not progress:
            break
    
    # Try a different greedy order
    # Greedy by value (smallest first)
    selected2 = set()
    for node in range(1, N + 1):
        if all(node not in conflicts[s] for s in selected2):
            selected2.add(node)
    
    if len(selected2) > len(best_set):
        best_set = set(selected2)
    
    # Try greedy by largest first
    selected3 = set()
    for node in range(N, 0, -1):
        if all(node not in conflicts[s] for s in selected3):
            selected3.add(node)
    
    if len(selected3) > len(best_set):
        best_set = set(selected3)
    
    # Simulated annealing / random restarts
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random greedy
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        random_set = set()
        for node in random_order:
            if all(node not in conflicts[s] for s in random_set):
                random_set.add(node)
        
        if len(random_set) > len(best_set):
            best_set = set(random_set)
        
        # Local search on random_set
        sa_set = set(random_set)
        for ls_iter in range(100):
            if time.time() > deadline - 5:
                break
            # Try to add any free node
            added = False
            for node in range(1, N + 1):
                if node not in sa_set and all(node not in conflicts[s] for s in sa_set):
                    sa_set.add(node)
                    added = True
            
            # Try 1-for-1 swaps
            for node in range(1, N + 1):
                if node in sa_set:
                    continue
                blockers = [s for s in sa_set if node in conflicts[s]]
                if len(blockers) == 1:
                    blocker = blockers[0]
                    if all(node not in conflicts[s] for s in sa_set if s != blocker):
                        # Neutral swap, skip to avoid cycles
                        pass
            
            if not added:
                break
        
        if len(sa_set) > len(best_set):
            best_set = set(sa_set)
    
    return list(best_set)

# EVOLVE-BLOCK-END
