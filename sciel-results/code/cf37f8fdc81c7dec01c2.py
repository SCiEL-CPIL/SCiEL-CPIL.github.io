p, q = [1], [1]
while len(p) < 24:
    p, q = p + q, p + [-c for c in q]
