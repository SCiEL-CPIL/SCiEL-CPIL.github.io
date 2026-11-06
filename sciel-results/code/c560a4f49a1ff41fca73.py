"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction

    target = Fraction(5, 31)
    LIMIT = 10_000_000

    # Beam search minimising the LARGEST denominator (not the term count).
    # Each beam member: (max_d, remainder, last_d, path)
    # We only ever add a strictly larger denominator, guaranteeing distinctness.
    K = 500      # beam width
    T = 15       # candidate denominators tried per member (smallest valid first)
    MAX_DEPTH = 256

    beam = [(0, target, 0, [])]
    best = None  # (max_d, path)

    for _ in range(MAX_DEPTH):
        if not beam:
            break
        nxt = []
        for md, rem, last, path in beam:
            if rem == 0:
                if best is None or md < best[0]:
                    best = (md, list(path))
                continue
            if rem < 0:
                continue
            # smallest d with 1/d <= rem  ->  d >= ceil(1/rem)
            d0 = (rem.denominator + rem.numerator - 1) // rem.numerator
            if d0 <= last:
                d0 = last + 1
            for d in range(d0, d0 + T):
                if d > LIMIT:
                    break
                if d <= last:
                    continue
                nr = rem - Fraction(1, d)
                if nr < 0:
                    continue
                nmd = md if md >= d else d
                if best is not None and nmd >= best[0]:
                    continue
                nxt.append((nmd, nr, d, path + [d]))
        if not nxt:
            break
        # keep the K most promising: smallest max denominator first,
        # then smallest remainder (closest to completion).
        nxt.sort(key=lambda x: (x[0], x[1]))
        beam = nxt[:K]
        # update best from any completed member
        for md, rem, last, path in beam:
            if rem == 0 and (best is None or md < best[0]):
                best = (md, list(path))
        if best is not None and beam[0][0] >= best[0]:
            break

    if best is not None:
        return best[1]

    # Fallback: exact greedy completion (always valid).
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
