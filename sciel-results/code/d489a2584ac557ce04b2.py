import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    P = 101
    D = 10
    elements = list(range(1, D + 1))
    
    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)
        
    # Precompute all non-empty subsets with their sums and sizes
    subsets = []  # list of (sum_mod_P, size, elements_tuple)
    for mask in range(1, 1 << D):
        res = 0
        elems = []
        sz = 0
        for i in range(D):
            if mask & (1 << i):
                x = i + 1
                res = (res + inv[x]) % P
                elems.append(x)
                sz += 1
        subsets.append((res, sz, tuple(elems)))
    
    # Group subsets by their sum residue
    residue_to_subsets = defaultdict(list)
    for res, sz, elems in subsets:
        residue_to_subsets[res].append((sz, elems))
    
    # Sort subsets in each residue group by size (smallest first)
    for res in residue_to_subsets:
        residue_to_subsets[res].sort(key=lambda x: x[0])
    
    # Check if all residues are covered
    uncovered = [r for r in range(P) if r not in residue_to_subsets]
    if uncovered:
        # This shouldn't happen, but return empty lists for uncovered residues
        result = []
        for r in range(P):
            if r in residue_to_subsets:
                result.append(list(residue_to_subsets[r][0][1]))
            else:
                result.append([])
        return result
    
    # Try to find a perfect matching that minimizes the maximum row length
    # We'll use a greedy approach: try max_size = 1, 2, 3, ... until we find a valid assignment
    # For each max_size, we can only use subsets with size <= max_size
    
    def can_assign(max_size):
        """Check if we can assign a subset to each residue using only subsets of size <= max_size.
        Returns the assignment if possible, None otherwise."""
        
        # Build adjacency: residue -> list of (subset_index, subset)
        # We'll use a simple greedy matching with backtracking for small cases
        # Since P=101 and subsets are limited, we can use a more direct approach
        
        # For each residue, filter subsets to those with size <= max_size
        residue_options = {}
        for r in range(P):
            if r in residue_to_subsets:
                options = [(sz, elems) for sz, elems in residue_to_subsets[r] if sz <= max_size]
                if options:
                    residue_options[r] = options
                else:
                    return None
            else:
                return None
        
        # Now we need to find a system of distinct representatives
        # Since denominators can recur across rows, this is actually just:
        # For each residue, pick any subset from its options.
        # Wait - the constraint is that denominators are distinct WITHIN a row,
        # but they CAN recur across different rows.
        # So there's no global constraint! Each residue just needs ANY valid subset.
        # The minimum max row length is just the minimum over all residues of
        # the minimum subset size for that residue.
        
        # Actually, re-reading the problem: "Denominators may recur across different rows, 
        # but never within one row."
        # This means each row independently has distinct denominators, but the same 
        # denominator can appear in multiple rows.
        # So the problem decomposes! For each residue r, we just need the smallest 
        # subset that sums to r mod 101.
        # The maximum row length is then max over all r of min_subset_size(r).
        
        # So the answer is simply: for each residue, take the smallest subset.
        # Let's compute that.
        
        assignment = {}
        for r in range(P):
            # Find the smallest subset for this residue
            min_sz = None
            min_elems = None
            for sz, elems in residue_to_subsets[r]:
                if sz <= max_size:
                    if min_sz is None or sz < min_sz:
                        min_sz = sz
                        min_elems = elems
                    if min_sz == 1:  # Can't do better
                        break
            if min_elems is None:
                return None
            assignment[r] = min_elems
        
        return assignment
    
    # Since the problem decomposes (no cross-row constraints on denominators),
    # the optimal solution is simply: for each residue, use the smallest subset.
    # The max row length is max over all residues of the min subset size.
    
    result = []
    max_row_len = 0
    for r in range(P):
        # Find the smallest subset for this residue
        min_sz = None
        min_elems = None
        for sz, elems in residue_to_subsets[r]:
            if min_sz is None or sz < min_sz:
                min_sz = sz
                min_elems = elems
            if min_sz == 1:
                break
        
        if min_elems is None:
            # Should not happen
            result.append([])
        else:
            result.append(list(min_elems))
            if len(min_elems) > max_row_len:
                max_row_len = len(min_elems)
    
    return result
