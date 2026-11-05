   min_d = min(_hamming(a, b) for a, b in itertools.combinations(code, 2) if a < b)
   