<<<<<<< SEARCH
            # So for the block-level, the W coefficients for product t in C block (bi, bk):
            wcoeff_block = {}
            for (bi, bk), coeff in [(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1]:
                pass
=======
            # So for the block-level, the W coefficients for product t in C block (bi, bk):
            # t=0 (P1): C11 += 1, C22 += 1
            # t=1 (P2): C12 += 1, C22 -= 1
            # t=2 (P3): C21 += 1, C22 += 1
            # t=3 (P4): C11 += 1, C22 += 1
            # t=4 (P5): C11 -= 1, C12 += 1
            # t=5 (P6): C21 += 1
            # t=6 (P7): C11 += 1
>>>>>>> REPLACE
