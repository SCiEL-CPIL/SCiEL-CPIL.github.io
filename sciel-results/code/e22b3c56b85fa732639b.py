    # After building t_to_blocks as list of lists:
    t_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_to_blocks]
    block_t_map_np = [np.array(lst, dtype=np.int32) for lst in block_t_map]
