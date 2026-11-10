    _TRIPLE_TO_BLOCKS = [[] for _ in range(_TOTAL_TRIPLES)]
    for b_idx, block in enumerate(_BLOCKS_LIST):
        for triple in itertools.combinations(block, T):
            tid = _TRIPLE_TO_ID[triple]
            _TRIPLE_TO_BLOCKS[tid].append(b_idx)
