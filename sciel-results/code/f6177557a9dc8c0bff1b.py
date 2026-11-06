    right_vals = [float(values[i]) for i in right_indices]
    right_subsets = [(0.0, 0)]
    for i, v in enumerate(right_vals):
        # Add v to all existing subsets
        # List comprehension
        # Note: This creates a new list.
        # To save memory, we can extend in place?
        # current_len = len(right_subsets)
        # for j in range(current_len):
        #    s, m = right_subsets[j]
        #    right_subsets.append((s+v, m | (1<<i)))
        # This is slow in Python loop.
        
        # List comprehension is faster:
        additions = [(s + v, m | (1 << i)) for s, m in right_subsets]
        right_subsets.extend(additions)
