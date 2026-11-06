import time
import itertools

def construct():
    P = 101
    D = 10
    elements = list(range(1, D + 1))
    
    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)
    
    # Enumerate all non-empty subsets and their sum mod P
    # For each residue, collect all subsets that sum to it
    subsets_by_res = [[] for _ in range(P)]
    
    for mask in range(1, 1 << D):
        res = 0
        elems = []
        for i in range(D):
            if mask & (1 << i):
                x = i + 1
                res = (res + inv[x]) % P
                elems.append(x)
        subsets_by_res[res].append((len(elems), elems))
    
    # For each residue, find the minimum-size subset
    # But we need to minimize the MAXIMUM row length across all residues
    # Since each residue can independently choose its best subset,
    # the optimal max row length is just the max over all residues of the min size for that residue
    
    # Find min size for each residue
    min_sizes = [float('inf')] * P
    best_elems = [None] * P
    for r in range(P):
        if subsets_by_res[r]:
            # Sort by size, take the smallest
            subsets_by_res[r].sort(key=lambda x: x[0])
            min_sizes[r] = subsets_by_res[r][0][0]
            best_elems[r] = subsets_by_res[r][0][1]
    
    # The answer is max(min_sizes)
    max_len = max(min_sizes)
    
    # Construct result
    result = []
    for r in range(P):
        if best_elems[r] is not None:
            result.append(best_elems[r])
        else:
            result.append([])
    
    return result
