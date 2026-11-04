    if _time_remaining() > 10.0:
        refined = _refine(U0, V0, W0, n, m, p)
        if refined is not None:
            U0, V0, W0 = refined
