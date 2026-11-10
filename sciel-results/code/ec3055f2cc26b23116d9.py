import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 10
    p, q = 2, 5

    def decomp(i):
        # i in 0..9 -> (a in Z2, b in Z5) where i = a*q + b
        return (i // q, i % q)

    def is_latin(sq):
        for i in range(n):
            row = sq[i]
            if len(set(row)) != n:
                return False
            col = [sq[j][i] for j in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(sq1, sq2):
        seen = set()
        for i in range(n):
            r1 = sq1[i]
            r2 = sq2[i]
            for j in range(n):
                pair = (r1[j], r2[j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Generate candidate Latin squares using algebraic constructions over Z_2 x Z_5
    candidates = []

    # Family 1: X = (a+c) mod 2, Y = (b + k*d + l*a*c) mod 5
    for k in range(5):
        for l in range(5):
            sq = [[0] * n for _ in range(n)]
            for i in range(n):
                a, b = decomp(i)
                for j in range(n):
                    c, d = decomp(j)
                    X = (a + c) % 2
                    Y = (b + k * d + l * a * c) % 5
                    sq[i][j] = 5 * X + Y
            candidates.append(sq)

    # Family 2: X = (a + c + r*b) mod 2, Y = (b + k*d + l*a) mod 5
    for r in range(2):
        for k in range(1, 5):
            for l in range(5):
                sq = [[0] * n for _ in range(n)]
                for i in range(n):
                    a, b = decomp(i)
                    for j in range(n):
                        c, d = decomp(j)
                        X = (a + c + r * b) % 2
                        Y = (b + k * d + l * a) % 5
                        sq[i][j] = 5 * X + Y
                candidates.append(sq)

    # Family 3: X = (a + c + r*b + s*d) mod 2, Y = (b + k*d + l*a + m*c) mod 5
    for r in range(2):
        for s in range(2):
            for k in range(1, 5):
                for l in range(5):
                    for m in range(5):
                        sq = [[0] * n for _ in range(n)]
                        for i in range(n):
                            a, b = decomp(i)
                            for j in range(n):
                                c, d = decomp(j)
                                X = (a + c + r * b + s * d) % 2
                                Y = (b + k * d + l * a + m * c) % 5
                                sq[i][j] = 5 * X + Y
                        candidates.append(sq)

    # Family 4: X = (a + c + r*b + s*d + t*a*d) mod 2, Y = (b + k*d + l*a + m*c + v*a*c) mod 5
    for r in range(2):
        for s in range(2):
            for t in range(2):
                for k in range(1, 5):
                    for l in range(5):
                        for m in range(5):
                            for v in range(5):
                                sq = [[0] * n for _ in range(n)]
                                for i in range(n):
                                    a, b = decomp(i)
                                    for j in range(n):
                                        c, d = decomp(j)
                                        X = (a + c + r * b + s * d + t * a * d) % 2
                                        Y = (b + k * d + l * a + m * c + v * a * c) % 5
                                        sq[i][j] = 5 * X + Y
                                candidates.append(sq)
                                if time.time() > deadline:
                                    break
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break_outer = True
    else:
        break_outer = False

    # Add some standard cyclic squares from inspiration
    if not break_outer:
        for shift in range(10):
            sq = [[(i + j + shift) % n for j in range(n)] for i in range(n)]
            candidates.append(sq)
        for shift in range(10):
            sq = [[(i - j + shift) % n for j in range(n)] for i in range(n)]
            candidates.append(sq)

    # Deduplicate and filter Latin
    seen_keys = set()
    valid = []
    for sq in candidates:
        if not is_latin(sq):
            continue
        key = tuple(tuple(row) for row in sq)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        valid.append(sq)

    # Greedily select mutually orthogonal set
    mols = []
    for sq in valid:
        if time.time() > deadline:
            break
        if all(are_orthogonal(sq, existing) for existing in mols):
            mols.append(sq)
        if len(mols) >= 9:
            break

    # If we didn't find enough, try a second pass with different ordering
    if len(mols) < 8 and time.time() < deadline:
        mols2 = []
        for sq in reversed(valid):
            if time.time() > deadline:
                break
            if all(are_orthogonal(sq, existing) for existing in mols2):
                mols2.append(sq)
            if len(mols2) > len(mols):
                mols = mols2
            if len(mols) >= 9:
                break

    # Ensure pure Python int types and list-of-lists
    out = []
    for sq in mols:
        out.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return out
