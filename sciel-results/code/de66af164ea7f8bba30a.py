    block_t_indices = []
    for block in all_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_sub_to_idx[sub])
        block_t_indices.append(indices)
    block_t_indices = np.array(block_t_indices, dtype=np.int32)
