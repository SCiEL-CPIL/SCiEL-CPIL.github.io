    def is_bad(c0, c1, c2, c3):
        # < 3 distinct means 1 or 2 distinct
        # 1 distinct
        if c0 == c1 == c2 == c3: return True
        # 2 distinct
        # This is true if (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        # AND not all same (already covered)
        # Wait, if c0==c1 and c2==c3, it could be c0==c2 (1 distinct) or c0!=c2 (2 distinct).
        # So:
        if (c0 == c1 and c2 == c3) or (c0 == c2 and c1 == c3) or (c0 == c3 and c1 == c2):
            return True
        return False
    