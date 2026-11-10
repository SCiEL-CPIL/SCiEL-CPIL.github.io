    def is_compatible(v, mask):
        for a, b in bad_pairs[v]:
            if (mask >> a) & 1 and (mask >> b) & 1:
                return False
        return True
    