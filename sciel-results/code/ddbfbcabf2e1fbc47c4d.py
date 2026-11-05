mask_to_verts = {}
for mask in all_subsets_8:
    verts = tuple(i for i in range(N) if (mask >> i) & 1)
    mask_to_verts[mask] = verts
