import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Precompute modular inverses for 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # best_rep[r] will store the smallest subset (as a tuple for hashing, list for return) 
    # that sums to r mod 101.
    # Initialize with None
    best_rep = [None] * 101
    
    # We want to minimize the maximum length.
    # So we should try to assign short subsets to all residues.
    # Iterate by subset size from 1 to 10.
    # For each size, iterate through all combinations of that size.
    # If a residue is not yet assigned, assign this subset.
    # This greedy approach ensures that if a residue can be covered by a subset of size k,
    # it gets covered by size k, not by a larger size found later.
    
    # Note: It is possible that some residues are not covered by small subsets.
    # The loop continues until all are covered or we run out of subsets.
    
    covered = [False] * 101
    remaining = 101
    
    # Generate subsets by size
    for size in range(1, 11):
        if time.time() > deadline:
            break
        if remaining == 0:
            break
            
        # Generate all combinations of {1..10} of given size
        for combo in combinations(range(1, 11), size):
            s = 0
            for x in combo:
                s += inv[x]
            r = s % 101
            
            if not covered[r]:
                best_rep[r] = list(combo)
                covered[r] = True
                remaining -= 1
                
                if remaining == 0:
                    break
                    
    # If some residues are still not covered (should not happen for this small problem),
    # fall back to any valid representation. 
    # But for p=101 and denom 1..10, all residues are covered.
    # Just to be safe, if any is None, we need to find a representation.
    # However, the problem states "Every residue must be represented".
    # The seed code implies it's always possible.
    
    # Final check and construction of result
    result = []
    for r in range(101):
        if best_rep[r] is None:
            # Fallback: find any subset for this residue
            # This shouldn't trigger for this specific instance
            found = False
            for size in range(1, 11):
                for combo in combinations(range(1, 11), size):
                    s = sum(inv[x] for x in combo) % 101
                    if s == r:
                        best_rep[r] = list(combo)
                        found = True
                        break
                if found:
                    break
        result.append(best_rep[r])
        
    return result
