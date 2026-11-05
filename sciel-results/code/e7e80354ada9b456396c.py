    # Strategy 4: XOR-based
    colors4 = np.array([((a ^ c) % 2) for a, b, c in triples], dtype=np.int8)
