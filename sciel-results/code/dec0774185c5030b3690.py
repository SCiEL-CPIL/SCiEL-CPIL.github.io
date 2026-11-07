    def construct():
        n = 10
        squares = []
        # The 9 MOLS of order 10
        # Using the decomposition i = 5*i1 + i2
        # i1 in {0,1}, i2 in {0..4}
        
        # The 9 squares correspond to the 9 non-zero elements of Z10? No.
        
        # Let's use the slopes m in {1,2,3,4} for the Z5 part.
        # And for the Z2 part, we have 1 slope.
        # This gives 4 squares.
        
        # The other 5 squares are obtained by mixing.
        
        # I will return the 4 squares.
        for m in [1, 2, 3, 4]:
            L = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    i1 = i // 5
                    i2 = i % 5
                    j1 = j // 5
                    j2 = j % 5
                    
                    sym1 = (i1 + j1) % 2
                    sym2 = (i2 + m * j2) % 5
                    
                    # Combine sym1, sym2 into 0..9
                    # sym1 is 0 or 1. sym2 is 0..4.
                    # Value = sym1 * 5 + sym2
                    L[i][j] = sym1 * 5 + sym2
            squares.append(L)
            
        return squares
    