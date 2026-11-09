def line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # A = dy, B = -dx, C = -(dy*x1 - dx*y1)
    a = dy
    b = -dx
    c = -(dy * x1 - dx * y1)
    g2 = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
        a, b, c = -a, -b, -c
    return (a, b, c)
