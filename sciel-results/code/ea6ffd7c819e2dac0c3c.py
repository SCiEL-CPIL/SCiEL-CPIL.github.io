import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    # Precompute K4s and their edge indices
    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    num_k4 = len(k4_edges)
    
    # For backtracking, we need to know which K4s an edge belongs to
    # and which other edges in that K4 are "prior" to it in our assignment order.
    # We assign edges in lexicographic order: (0,1), (0,2), ..., (0,15), (1,2), ...
    # This is the natural order of the list of 120 edges.
    
    # Map edge index to list of K4 indices it participates in
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Precompute for each K4, the set of edges.
    # When assigning edge e, we can check K4s where e is the LAST edge to be assigned.
    # A K4 is "complete" when all its 6 edges are assigned.
    # In lexicographic edge order, a K4 {a,b,c,d} (a<b<c<d) is completed when 
    # we assign the edge (c,d), because (c,d) is the last edge in that K4 to appear
    # in the lexicographic sequence of all edges.
    # Let's verify:
    # Edges in K4(a,b,c,d): (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
    # Lex order: (a,b) < (a,c) < (a,d) < (b,c) < (b,d) < (c,d).
    # Yes, (c,d) is the last one.
    
    # So, for each edge e, we can precompute the list of K4s where e is the "completing" edge.
    # This allows O(1) or small constant check per assignment instead of checking all K4s.
    
    edge_completes_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        # Find the edge with the maximum index in this K4? 
        # Actually, the "last" edge in lex order is (max1, max2) where max1 < max2 are the two largest vertices.
        # Let's just find the max edge index in the list.
        last_e = max(edges)
        edge_completes_k4s[last_e].append(k4_idx)
        
    # State for backtracking
    coloring = [-1] * num_edges
    # distinct_count[k4_idx] = number of distinct colors so far? 
    # No, we need to know the set of colors. Since max colors is small (<=15), 
    # we can use a bitmask for each K4.
    k4_masks = [0] * num_k4
    
    # We will try to find a valid coloring with k colors.
    # Start with k=13 (lower bound), if fails, try 14, then 15.
    
    def backtrack(edge_idx, k):
        """
        Recursive backtracking.
        Returns True if a valid coloring is found.
        Modifies 'coloring' and 'k4_masks' in place.
        """
        if time.time() > deadline:
            return False
            
        if edge_idx == num_edges:
            return True
            
        # Try each color in 0..k-1
        # Randomize color order to avoid systematic bias? 
        # Or use a heuristic? 
        # For now, try 0..k-1. 
        # To speed up, we can try colors that are already used in adjacent edges first?
        # But with small k, 0..k-1 is fine.
        
        # Optimization: Check if any K4 completed by this edge would violate constraint
        # for each color.
        
        # Get the K4s that this edge completes
        completed_k4s = edge_completes_k4s[edge_idx]
        
        # If no K4 is completed, we don't need to check distinct count immediately,
        # but we still need to update the mask for future checks.
        
        # Try colors
        for c in range(k):
            coloring[edge_idx] = c
            bitmask = 1 << c
            
            # Update masks for K4s that include this edge
            # We only need to check validity for K4s that are COMPLETED by this edge.
            # For K4s that are not yet complete, we just update the mask.
            
            valid = True
            # We need to update masks for ALL K4s this edge belongs to, 
            # because future edges in those K4s will need the current mask state.
            
            # However, checking validity only happens when the K4 is complete.
            
            # To avoid updating and reverting masks for all K4s on every color try,
            # we can be smart. But with 120 edges and ~10 K4s per edge, it's manageable.
            
            # List of K4s affected
            affected_k4s = edge_to_k4s[edge_idx]
            
            # Save old masks to revert if we backtrack
            old_masks = [k4_masks[k4] for k4 in affected_k4s]
            
            # Apply new color
            for k4 in affected_k4s:
                k4_masks[k4] |= bitmask
                
            # Check validity for completed K4s
            for k4 in completed_k4s:
                # Count bits in mask
                if bin(k4_masks[k4]).count('1') < 5:
                    valid = False
                    break
            
            if valid:
                if backtrack(edge_idx + 1, k):
                    return True
            else:
                pass
            
            # Revert masks
            for i, k4 in enumerate(affected_k4s):
                k4_masks[k4] = old_masks[i]
                
        coloring[edge_idx] = -1
        return False

    # Try k=13, 14, 15
    for k in [13, 14, 15]:
        # Reset state
        coloring = [-1] * num_edges
        k4_masks = [0] * num_k4
        
        # Add a timeout check in the loop? 
        # The backtrack function checks time at each step.
        
        if backtrack(0, k):
            # Found a valid solution with k colors
            # Return the coloring
            # Ensure all colors are in 0..k-1 (they are by construction)
            return coloring
            
        if time.time() > deadline - 5.0:
            break

    # If backtracking fails or times out, fallback to a known valid solution
    # (i+j)%17 uses 17 colors. But we want to minimize.
    # If we didn't find 13-15, maybe we can return a 15-color solution if we have one?
    # The backtracker might have found a partial one? No, it returns None/False.
    
    # Fallback: Random search with SA as in seed, but faster?
    # Or just return the (i+j)%17 solution which is valid (uses 17 colors, score 17/seed_value).
    # Seed value is likely around 15-17. If seed is 15, 17 is worse.
    # But better than invalid (0).
    
    # Let's try a quick greedy/random fill if time permits
    if time.time() < deadline - 2.0:
         # Simple random restart with few steps
         for _ in range(10):
             if time.time() > deadline - 1.0: break
             coloring = [random.randint(0, 12) for _ in range(num_edges)]
             # Check validity
             valid = True
             for edges in k4_edges:
                 cols = set(coloring[e] for e in edges)
                 if len(cols) < 5:
                     valid = False
                     break
             if valid:
                 return coloring
                 
    # Last resort: (i+j)%17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# Keep the function name and signature exactly as required
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program has: def construct():
# So we are good.

# To ensure we don't hit recursion limit (120 depth is fine, but let's be safe)
sys.setrecursionlimit(10000)

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors used: {len(set(res))}")
    # Verify
    n = 16
    def get_idx(u, v):
        if u > v: u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
    ok = True
    for a,b,c,d in itertools.combinations(range(n), 4):
        edges = [get_idx(a,b), get_idx(a,c), get_idx(a,d), get_idx(b,c), get_idx(b,d), get_idx(c,d)]
        if len(set(res[e] for e in edges)) < 5:
            ok = False
            break
    print(f"Valid: {ok}")
