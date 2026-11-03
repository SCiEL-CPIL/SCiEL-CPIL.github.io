    block_tsets_indices = []
    for block in all_blocks:
        indices = [tset_index[ts] for ts in itertools.combinations(block, t)]
        block_tsets_indices.append(indices)
