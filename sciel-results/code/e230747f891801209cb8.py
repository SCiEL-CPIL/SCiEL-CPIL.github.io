for rest in itertools.combinations(block_set, t - 1):
    ts_tuple = tuple(sorted(rest + (p,)))
    ts_idx = tsubset_index[ts_tuple]
    if is_uncovered[ts_idx]:
        gain += 1
