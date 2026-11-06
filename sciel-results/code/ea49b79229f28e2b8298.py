def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    if not selected:
        return True
    
    n = len(selected)
    
    # Case 1: x is the divisor.
    # Check if any triple from selected sums to a multiple of x.
    # (sel[i] + sel[j] + sel[k]) % x == 0
    # Equivalent to: exists i, j, k distinct such that sel[k] % x == (-sel[i] - sel[j]) % x
    
    # Precompute residues of selected elements modulo x
    residues = [v % x for v in selected]
    # Count frequencies of residues
    from collections import Counter
    res_counts = Counter(residues)
    
    # We need to find i, j, k distinct.
    # Iterate pairs (i, j) from selected.
    # For each pair, required residue r = (-selected[i] - selected[j]) % x
    # We need a k such that residues[k] == r and k != i and k != j.
    
    # To handle distinctness efficiently:
    # If res_counts[r] >= 3, we can always find 3 distinct? No, we need specific i, j.
    # Let's iterate i, j.
    # For a fixed i, j, we need k != i, j with residues[k] == r.
    # The number of available k's with residue r is res_counts[r].
    # If i or j has residue r, we must subtract them from the count.
    
    for i in range(n):
        for j in range(i + 1, n):
            r = (-selected[i] - selected[j]) % x
            count_r = res_counts.get(r, 0)
            # Subtract i and j if they have residue r
            if residues[i] == r:
                count_r -= 1
            if residues[j] == r:
                count_r -= 1
            if count_r > 0:
                return False # Violation found: x divides sum of i, j, k
    
    # Case 2: a in selected is the divisor.
    # For each a in selected, check if there exist i, j in selected \ {a}
    # such that (sel[i] + sel[j] + x) % a == 0.
    # Equivalent to: exists i, j distinct, != a_idx, such that (sel[i] + sel[j]) % a == (-x) % a
    
    for idx_a, a in enumerate(selected):
        target = (-x) % a
        # We need pair (i, j) from selected excluding idx_a
        # such that (sel[i] + sel[j]) % a == target
        
        # Precompute residues of selected \ {a} modulo a?
        # Or just iterate pairs? N is small.
        # O(N^2) per a is O(N^3) total.
        # Optimization: Use residue counts again.
        res_counts_a = Counter()
        for idx in range(n):
            if idx == idx_a:
                continue
            res_counts_a[selected[idx] % a] += 1
        
        # Now iterate i from selected \ {a}
        # We need j != i, j != a such that (sel[i] + sel[j]) % a == target
        # => sel[j] % a == (target - sel[i]) % a
        
        for idx_i in range(n):
            if idx_i == idx_a:
                continue
            r_i = selected[idx_i] % a
            r_needed = (target - r_i) % a
            count_needed = res_counts_a.get(r_needed, 0)
            
            # Subtract idx_i if it has the needed residue
            if r_i == r_needed:
                count_needed -= 1
            
            if count_needed > 0:
                return False # Violation found: a divides sel[i] + sel[j] + x
                
    return True
