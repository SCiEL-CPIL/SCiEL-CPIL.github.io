def construct():
    import time
    deadline = time.time() + 80.0   # safety margin below the 90 s wall-clock limit
    p = 1009

    # Bitset of non-zero quadratic residues: bit v set iff v is a QR mod p, v != 0.
    # Also record one root for each residue (first root encountered).
    mask = 0
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v == 0:
            continue
        mask |= (1 << v)
        if v not in roots:
            roots[v] = x

    # Windowed mask: five consecutive 1-bits.  For a candidate start s,
    # (mask & (window << s)) != 0  <=>  s..s+4 are all non-zero QRs.
    window = (1 << 5) - 1  # 0b11111

    best = None
    # Ascending scan finds the MINIMUM start directly; the expected answer
    # appears within the first few dozen positions (QR density ~0.5).
    for s in range(1, 1005):
        if (mask & (window << s)) != 0:
            best = (s, [roots[s + i] for i in range(5)])
            break
        if time.time() > deadline:
            break

    if best is None:
        # Fallback (should not occur): verify with explicit checks, ascending.
        for s in range(1, 1005):
            if all((roots.get(s + i) is not None) for i in range(5)):
                best = (s, [roots[s + i] for i in range(5)])
                break
        if best is None:
            raise RuntimeError('No run of five consecutive non-zero QRs found')

    s, rts = best
    # Certification: each root squared must equal the corresponding residue.
    for i, r in enumerate(rts):
        assert (r * r) % p == s + i, "root certification failed"
    assert 1 <= s <= 1004 and all(1 <= r <= 1008 for r in rts)

    return {'start': s, 'roots': rts}
