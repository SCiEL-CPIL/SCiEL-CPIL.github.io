def _minimize_1d(obj, a, b, steps=20):
    for _ in range(steps):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1 = obj(m1)
        f2 = obj(m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
    return (a + b) / 2.0
