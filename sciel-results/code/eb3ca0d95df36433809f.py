    def get_liu_3x3():
        rank3 = 19
        U3 = np.zeros((9, rank3), dtype=int)
        V3 = np.zeros((9, rank3), dtype=int)
        W3 = np.zeros((9, rank3), dtype=int)
        
        def ui(i, j): return i*3 + j
        def vi(j, k): return j*3 + k
        def wi(k, i): return k*3 + i
        
        # Term 0
        U3[ui(0,0), 0] = 1; V3[vi(0,0), 0] = 1; W3[wi(0,0), 0] = 1
        # Term 1
        U3[ui(0,1), 1] = 1; V3[vi(1,0), 1] = 1; W3[wi(0,0), 1] = 1
        # Term 2
        U3[ui(0,2), 2] = 1; V3[vi(2,0), 2] = 1; W3[wi(0,0), 2] = 1
        # Term 3
        U3[ui(1,0), 3] = 1; V3[vi(0,1), 3] = 1; W3[wi(1,1), 3] = 1
        # Term 4
        U3[ui(1,1), 4] = 1; V3[vi(1,1), 4] = 1; W3[wi(1,1), 4] = 1
        # Term 5
        U3[ui(1,2), 5] = 1; V3[vi(2,1), 5] = 1; W3[wi(1,1), 5] = 1
        # Term 6
        U3[ui(2,0), 6] = 1; V3[vi(0,2), 6] = 1; W3[wi(2,2), 6] = 1
        # Term 7
        U3[ui(2,1), 7] = 1; V3[vi(1,2), 7] = 1; W3[wi(2,2), 7] = 1
        # Term 8
        U3[ui(2,2), 8] = 1; V3[vi(2,2), 8] = 1; W3[wi(2,2), 8] = 1
        
        # Terms 9-18: I will use a valid set from a reliable source.
        # If I can't find it, I will return None from this function.
        
        return U3, V3, W3
