def _unembed2(arr, n, i):
    # arr is (2, 2^{n-1})
    # Reshape to (2, 2, ..., 2) with axis 0 being the moved one?
    # No, arr is (2, rest).
    # We need to go back to (2^n,) with axis i at position i.
    # Current shape: (2, 2^{n-1}).
    # Reshape to (2, 2, ..., 2) where the first 2 is axis i.
    # The rest axes are 0..n-1 excluding i, in order.
    # So shape is [2] * n.
    # But the axis i is at position 0 in the current tensor.
    # We need to move it back to position i.
    # perm_inv: move 0 to i.
    # New perm: [0, 1, ..., i-1, i, i+1, ...] ?
    # Let's just use transpose.
    # Current axes: [i, 0, 1, ..., i-1, i+1, ...] (logical)
    # We want: [0, 1, ..., i-1, i, i+1, ...]
    # So we need to move axis 0 to position i.
    # This is equivalent to transpose with:
    # new_axis_0 = old_axis_1
    # ...
    # new_axis_i = old_axis_0
    # ...
    # Let's construct the perm.
    # The current tensor has shape (2, 2, ..., 2).
    # Let's call the current axes 0, 1, ..., n-1.
    # Axis 0 corresponds to qubit i.
    # Axis 1 corresponds to qubit 0 (if i!=0) or 1 (if i==0)?
    # In _embed2, perm was [i] + [j for j in range(n) if j != i].
    # So current axis 0 is old i.
    # Current axis k (k>0) is old (perm[k]).
    # To invert, we need to map current axes to old axes.
    # old_axis = perm_inv[current_axis]
    # perm_inv is the inverse permutation of perm.
    perm = [i] + [j for j in range(n) if j != i]
    # Inverse perm:
    inv_perm = [0] * n
    for k, p in enumerate(perm):
        inv_perm[p] = k
    # We want to transpose such that the new axis k is the old axis inv_perm[k]?
    # No, we want to restore the original order.
    # The current tensor T has T[a_0, ..., a_{n-1}] = psi[a_{perm[0]}, ..., a_{perm[n-1]}]
    # We want P[b_0, ..., b_{n-1}] such that P = psi.
    # So P[b] = T[inv_perm[b]]?
    # Actually, just transpose T by inv_perm.
    # T.transpose(inv_perm) will have axis k = T axis inv_perm[k].
    # Let's check:
    # T is (2^n,).
    # T.reshape([2]*n).transpose(inv_perm)
    # Then flatten.
    return arr.reshape([2]*n).transpose(inv_perm).reshape(-1)
