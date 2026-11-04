    def mul_2x4x4(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1):
        # Left: 2x4 x 4x2 (B cols qk0, qk1)
        mul_2x4x2(pi0, pi1, pj0, pj0+1, qj0, qj0+1, qk0, qk1)
        # Right: 2x4 x 4x2 (B cols qk0+2, qk1+2)
        mul_2x4x2(pi0, pi1, pj0, pj0+1, qj0, qj0+1, qk0+2, qk1+2)
