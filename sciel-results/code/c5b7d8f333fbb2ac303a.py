    cmp_bits = [0] * N
    for a in range(N):
        bits = 0
        # Subsets
        sub = a
        while sub:
            if sub != a:
                bits |= (1 << sub)
            sub = (sub - 1) & a
        # Supersets
        # Supersets of a are b such that b & a == a.
        # Iterate over complements?
        # Let c = ~a & (N-1).
        # b = a | s where s is subset of c.
        c = (~a) & (N - 1)
        sub = c
        while True:
            b = a | sub
            if b != a:
                bits |= (1 << b)
            if sub == 0: break
            sub = (sub - 1) & c
            
        cmp_bits[a] = bits
