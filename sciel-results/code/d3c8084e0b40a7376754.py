    def vec9_u(pairs):
        v = [0] * 9
        for a, b, s in pairs:
            v[a * 3 + b] = s
        return v
