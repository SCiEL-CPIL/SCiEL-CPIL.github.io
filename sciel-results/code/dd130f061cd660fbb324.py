        if 2 * x in S:
            return False
        for a in S:
            if x + a in S:
                return False
            if x - a in S:
                return False
        return True
