import time
import random
import numpy as np

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 88.0
    
    # Precompute all valid triples (a, b, c) with a < b < c
    # such that a+b, a+c, b+c are all <= 2*n (max possible sum)
    # Actually, we just need to check if sums are in the set.
    # A set S is valid if for no triple (a,b,c) in S, all three sums are in S.
    
    # To speed up checking, we can represent S as a boolean array of size 2*n+1 (indices 0..2n)
    # But membership check in Python set is O(1) average.
    # The bottleneck is the O(N^3) loop.
    # N is at most 128. C(128,3) ~ 340k.
    # 340k * 3 set lookups = ~1M ops per check.
    # In Python, 1M ops takes ~0.05s.
    # So we can do ~2000 checks per second.
    # In 85s, we can do ~170,000 checks.
    
    # Strategy:
    # 1. Generate random sets of size ~60-80.
    # 2. Check validity.
    # 3. If valid, try to add elements (greedy) to maximize size.
    # 4. If invalid, try to remove elements that are part of "bad" triples.
    
    # Better Strategy:
    # Start with a known large valid set (e.g., top half).
    # Then perform random swaps: remove one, add one, or remove one, or add one.
    # Keep track of best valid set found.
    
    # Optimization:
    # Represent set as a sorted list.
    # To check validity efficiently, we can precompute "conflict triples".
    # A triple (a,b,c) conflicts if a+b, a+c, b+c are all in S.
    # Instead of checking all triples, we can check for each pair (a,b) in S,
    # if a+b is in S, then we have a pair (a,b) whose sum is in S.
    # Let Pairs = {(a,b) in S | a<b and a+b in S}.
    # If there is a triple (a,b,c) such that (a,b), (a,c), (b,c) are all in Pairs, it's invalid.
    # This is a graph problem: Nodes are elements of S. Edges are pairs whose sum is in S.
    # We want to ensure there is no triangle (K3) in this graph.
    # So the graph of "sum-in-S" pairs must be triangle-free.
    
    # Algorithm:
    # 1. Maintain S as a set.
    # 2. Build graph G where edge (u,v) exists if u,v in S and u+v in S.
    # 3. Check if G has a triangle.
    #    If N is small, we can check triangles in O(N^2 * avg_degree) or similar.
    #    Or simply: for each node u, for each neighbor v, for each neighbor w of v (w > v),
    #    check if (u,w) is an edge.
    
    def has_triangle(S):
        S_list = sorted(S)
        S_set = set(S_list)
        # Build adjacency list
        # Only consider edges (u,v) with u < v
        adj = {x: [] for x in S_list}
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if u + v in S_set:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Check for triangles
        # For each u, for each v in adj[u], for each w in adj[v]:
        # if w in adj[u] and w > u (to avoid duplicates and self-loops)
        # Actually, just check if (u,w) is an edge.
        for u in S_list:
            for v in adj[u]:
                if v <= u: continue # only process edge once? No, adj is symmetric.
                for w in adj[v]:
                    if w > u and w in adj[u]:
                        return True
        return False

    # This is still potentially slow if degrees are high.
    # But max degree is bounded by N.
    
    # Let's try a different angle.
    # The set {63..128} is valid. Size 66.
    # Can we do better?
    # If we take {1..128}, it's definitely invalid.
    # What if we take all numbers > 64? Size 64.
    # What if we take all numbers in [43, 128]?
    # Min sum: 43+44=87. Max sum: 127+128=255.
    # If all sums are > 128, they are not in S.
    # So if min(S) + second_min(S) > max(S), it's valid.
    # For {43..128}, 43+44=87 < 128. So sums CAN be in S.
    # But we need a triple.
    
    # Let's use the graph triangle-free property.
    # We want a large subset S of {1..128} such that the graph G_S (edges where sum in S) is triangle-free.
    
    best_size = 0
    best_set = None
    
    # Initial guess: top half
    initial_set = set(range(65, 129)) # 65..128, size 64
    # Check if valid
    if not has_triangle(initial_set):
        best_size = len(initial_set)
        best_set = set(initial_set)
    else:
        # Fallback
        initial_set = set(range(63, 129))
        if not has_triangle(initial_set):
            best_size = len(initial_set)
            best_set = set(initial_set)
        else:
            # Just pick a small valid set
            initial_set = set(range(100, 129))
            best_size = len(initial_set)
            best_set = set(initial_set)

    current_set = set(best_set)
    
    # Local Search
    # Moves:
    # 1. Add a random element not in S.
    # 2. Remove a random element from S.
    # 3. Swap: remove one, add one.
    
    # To speed up, we can cache validity? No, set changes.
    
    # Let's try to add elements greedily first.
    # Sort candidates by value?
    
    # Randomized Hill Climbing
    for _ in range(20000): # Max iterations, controlled by time
        if time.time() > deadline:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add
            candidates = [x for x in range(1, 129) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = current_set | {x}
            if not has_triangle(new_set):
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            else:
                # Revert (do nothing)
                pass
                
        elif move_type < 0.8:
            # Try to remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            # Removing always keeps it valid if it was valid?
            # Yes, removing nodes cannot create a new triangle.
            # So if current_set is valid, new_set is valid.
            # But we want to maximize size, so only remove if we plan to add something else?
            # Or just remove to escape local maxima?
            # If we remove, size decreases. We only accept if it leads to a better state later?
            # Simple hill climbing: only accept if size increases.
            # So removal alone is never accepted unless we allow temporary decrease (Simulated Annealing).
            # Let's stick to Hill Climbing for simplicity and speed.
            # So we only do swaps or adds.
            pass
            
        else:
            # Swap
            if not current_set:
                continue
            x = random.choice(list(current_set))
            candidates = [y for y in range(1, 129) if y not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            new_set = (current_set - {x}) | {y}
            if not has_triangle(new_set):
                # Size is same. Do we accept?
                # It might allow future additions.
                # Accept with some probability?
                # For now, accept if it's "better" in some heuristic?
                # Or just accept to explore.
                # But we want to track best size.
                # Let's accept swaps to move around the landscape.
                current_set = new_set
                # Check if this allows immediate addition?
                # Maybe try to add after swap?
    
    # Final Greedy Addition
    # Try adding any remaining element
    for x in range(1, 129):
        if x not in current_set:
            new_set = current_set | {x}
            if not has_triangle(new_set):
                current_set = new_set
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
        
    return list(best_set)
