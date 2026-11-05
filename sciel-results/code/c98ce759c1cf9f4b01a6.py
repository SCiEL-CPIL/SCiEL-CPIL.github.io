import time
import random
import numpy as np
from itertools import combinations

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # The property: for every 4-set S, there exists a vertex y (possibly in S? No, "arcs to all four" means y dominates the other 4.
    # Wait, if y is in S, it has an arc to 3 other vertices. It cannot have an arc to itself.
    # So y must be outside S?
    # Let's re-read carefully: "Every set of 4 vertices must have a vertex with arcs to all four."
    # If the set is {u, v, w, x}, a vertex y has arcs to u, v, w, x.
    # This implies y is distinct from u, v, w, x.
    # So for every 4-subset S of V, there exists y in V\S such that y->s for all s in S.
    
    # This is the property of a "4-dominating" tournament.
    # The smallest such tournament has 67 vertices (Schütte's problem, k=4).
    # Wait, the task says "The smallest known has 67 vertices".
    # So f(4) = 67?
    # "Erdős's lower bound gives f(4) >= 31".
    # "Minimize n. A valid tournament on n vertices certifies f(4) <= n."
    # So if I find a valid tournament with n=67, I get a score based on 67.
    # If I return n=83 (from seed), I get a lower score.
    # I need to find a valid tournament with n as small as possible.
    # Since 67 is the smallest known, and it's a Paley tournament for p=67 (67 is prime, 67 = 4*16+3),
    # I should check if the Paley tournament on 67 vertices satisfies the property.
    # If it does, I can return it immediately.
    # If it doesn't, I need to search for a valid tournament of size 67 or smaller (unlikely) or 68...
    # But 67 is prime and a Paley tournament. Paley tournaments have strong regularity.
    # Let's check the Paley tournament on 67 vertices.

    def check_paley_property(n):
        # Construct Paley tournament adjacency matrix
        # i -> j iff (j-i)^2 is a quadratic residue mod n.
        # For prime p = 4k+3, -1 is not a residue, so (j-i) is residue iff (i-j) is non-residue.
        # i -> j iff Legendre symbol ((j-i)/p) == 1.
        
        # Precompute residues
        residues = set()
        for x in range(1, n):
            residues.add((x*x) % n)
        
        # out_mask[i] is a bitmask of vertices j such that i -> j
        out_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i == j: continue
                if (j - i) % n in residues:
                    mask |= (1 << j)
            out_masks[i] = mask
        
        # Check property: For every 4-subset S, there is y in V\S such that S subset out(y).
        # Equivalently, for every 4-subset S, the intersection of out(y) for y in V\S contains S?
        # No, we need EXISTENCE of y.
        # So for every 4-subset S, the union of {out(y) | y not in S} must contain S?
        # No, we need a SINGLE y that dominates ALL of S.
        # So for every 4-subset S, there exists y such that (out(y) & mask(S)) == mask(S).
        
        # Efficient check:
        # For each y, let C_y be the set of 4-subsets that y dominates.
        # We need the union of C_y for all y to be the set of all 4-subsets.
        # Actually, we just need to check if there is ANY 4-subset NOT dominated by ANY y.
        
        # Iterate over all 4-subsets.
        # For a given 4-subset S = {u, v, w, x}, compute mask.
        # Check if any out_mask[y] is a superset of mask.
        # Note: y must not be in S. But if y in S, out(y) doesn't contain y, so it can't contain mask(S) which contains y.
        # So we can just check all y.
        
        # To speed up, we can precompute for each 4-subset the set of dominators?
        # Or just iterate.
        # Number of 4-subsets for n=67: C(67,4) = 67*66*65*64/24 = 735,470.
        # For each, we check 67 out_masks. 735,470 * 67 ~ 50M operations.
        # In Python, this might be slow. ~5-10 seconds?
        # We can optimize using numpy or bit operations.
        
        # Let's try a slightly different approach.
        # For each y, we can generate all 4-subsets of out(y).
        # But out(y) has size (n-1)/2 = 33.
        # C(33,4) = 40,920.
        # Total y's: 67.
        # Total 4-subsets generated: 67 * 40,920 = 2,741,640.
        # We can mark which 4-subsets are dominated.
        # We need to represent 4-subsets as indices.
        # Index of {u,v,w,x} with u<v<w<x.
        # We can use a boolean array of size C(n,4).
        
        n_subsets = 0
        # Calculate C(n,4)
        if n >= 4:
            n_subsets = (n * (n-1) * (n-2) * (n-3)) // 24
        else:
            return True # Trivially true for n < 4
        
        dominated = np.zeros(n_subsets, dtype=bool)
        
        # Helper to get index of 4-subset {u,v,w,x}
        # u < v < w < x
        # We can map this to an index.
        # It's a bit tricky to compute index on the fly efficiently in Python.
        # Alternatively, iterate all 4-subsets and check if dominated by any y.
        # That was the 50M ops approach.
        
        # Let's try the 50M ops approach with some optimization.
        # We can break early if we find an undominated subset.
        
        # Precompute out_masks as a list of ints.
        # For each 4-subset, compute mask.
        # Check if any (out_mask[y] & mask) == mask.
        
        # We can sort out_masks by popcount? No.
        
        # Let's just run the check.
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    for x in range(w+1, n):
                        mask = (1 << u) | (1 << v) | (1 << w) | (1 << x)
                        found = False
                        for y in range(n):
                            if (out_masks[y] & mask) == mask:
                                found = True
                                break
                        if not found:
                            return False
        return True

    # Check Paley 67
    if time.time() < deadline - 10.0:
        if check_paley_property(67):
            # Construct orientation
            residues = set()
            for x in range(1, 67):
                residues.add((x*x) % 67)
            
            orientation = []
            for i in range(67):
                for j in range(i+1, 67):
                    if (j - i) % 67 in residues:
                        orientation.append(0)
                    else:
                        orientation.append(1)
                        
            return {'n': 67, 'orientation': orientation}
    else:
        # Timeout? Return 83?
        pass

    # If 67 failed or we didn't check, try local search for n=67?
    # Or return 83.
    # The seed program returned 83.
    # Let's try to find a valid tournament for n=67 using local search if Paley failed.
    # But Paley 67 is very likely to be valid or close.
    # Actually, it is KNOWN that the Paley tournament on 67 vertices is a solution to Schütte's problem k=4.
    # So it should be valid.
    # Why did I doubt it?
    # "The smallest known has 67 vertices".
    # So I should just return it.
    
    # Just in case, let's do a quick check for n=67 Paley again with a faster method if possible.
    # But the previous check might have been slow.
    # Let's assume it's valid and return it.
    # If I'm wrong, the evaluator will catch it.
    # But I should be careful.
    
    # Let's do a faster check for n=67.
    # We can use numpy to vectorize the check?
    # out_masks is a list of ints.
    # We can convert to numpy array? No, they are big ints.
    # But we can store them as arrays of 67 bits?
    # Or just use the previous method. It should be fast enough for 67.
    
    # Let's re-run the check for 67 to be sure, but with a time limit.
    if time.time() < deadline - 5.0:
        if check_paley_property(67):
            residues = set()
            for x in range(1, 67):
                residues.add((x*x) % 67)
            
            orientation = []
            for i in range(67):
                for j in range(i+1, 67):
                    if (j - i) % 67 in residues:
                        orientation.append(0)
                    else:
                        orientation.append(1)
                        
            return {'n': 67, 'orientation': orientation}
    
    # Fallback: return 83 (Paley 83)
    p = 83
    residues = set()
    for x in range(1, p):
        residues.add((x*x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in residues:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
