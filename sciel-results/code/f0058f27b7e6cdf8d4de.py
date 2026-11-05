        def get_strassen_2x2x2():
            # Returns U, V, W of shape (4, 7)
            # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
            u = [[0]*7 for _ in range(4)]
            v = [[0]*7 for _ in range(4)]
            w = [[0]*7 for _ in range(4)]
            
            # M0: (A00+A11)(B00+B11)
            u[0][0] = 1; u[3][0] = 1
            v[0][0] = 1; v[3][0] = 1
            w[0][0] = 1; w[3][0] = 1
            
            # M1: (A10+A11)B00
            u[2][1] = 1; u[3][1] = 1
            v[0][1] = 1
            w[2][1] = 1
            
            # M2: A00(B01-B11)
            u[0][2] = 1
            v[1][2] = 1; v[3][2] = -1
            w[1][2] = 1
            
            # M3: A11(B10-B00)
            u[3][3] = 1
            v[2][3] = 1; v[0][3] = -1
            w[2][3] = 1; w[3][3] = 1
            
            # M4: (A00+A01)B11
            u[0][4] = 1; u[1][4] = 1
            v[3][4] = 1
            w[3][4] = -1 # Wait, C11 = M0 - M1 + M2 + M5?
            # Let's re-verify Strassen formulas.
            # Standard:
            # p1 = (a00+a11)(b00+b11)
            # p2 = (a10+a11)b00
            # p3 = a00(b01-b11)
            # p4 = a11(b10-b00)
            # p5 = (a00+a01)b11
            # p6 = (a10-a00)(b00+b01)
            # p7 = (a01-a11)(b10+b11)
            #
            # c00 = p1 - p4 + p5 - p7
            # c01 = p3 + p5
            # c10 = p2 + p4
            # c11 = p1 - p2 + p3 + p6
            #
            # My indices:
            # M0 -> p1
            # M1 -> p2
            # M2 -> p3
            # M3 -> p4
            # M4 -> p5
            # M5 -> p6
            # M6 -> p7
            #
            # c00 = M0 - M3 + M4 - M6
            # c01 = M2 + M4
            # c10 = M1 + M3
            # c11 = M0 - M1 + M2 + M5
            #
            # Let's check W for M4 (p5).
            # p5 contributes to c00 (+1), c01 (+1), c11 (-1).
            # So w[0][4]=1, w[1][4]=1, w[3][4]=-1.
            # My code above had w[3][4]=-1 only. I need to add the others.
            
            # Let's rewrite the 2x2x2 generator carefully.
            
            u = [[0]*7 for _ in range(4)]
            v = [[0]*7 for _ in range(4)]
            w = [[0]*7 for _ in range(4)]
            
            # M0 (p1)
            u[0][0] = 1; u[3][0] = 1
            v[0][0] = 1; v[3][0] = 1
            w[0][0] = 1; w[3][0] = 1
            
            # M1 (p2)
            u[2][1] = 1; u[3][1] = 1
            v[0][1] = 1
            w[2][1] = 1
            
            # M2 (p3)
            u[0][2] = 1
            v[1][2] = 1; v[3][2] = -1
            w[1][2] = 1
            
            # M3 (p4)
            u[3][3] = 1
            v[2][3] = 1; v[0][3] = -1
            w[2][3] = 1; w[3][3] = 1
            
            # M4 (p5)
            u[0][4] = 1; u[1][4] = 1
            v[3][4] = 1
            w[0][4] = 1; w[1][4] = 1; w[3][4] = -1
            
            # M5 (p6)
            u[2][5] = 1; u[0][5] = -1
            v[0][5] = 1; v[1][5] = 1
            w[3][5] = 1
            
            # M6 (p7)
            u[1][6] = 1; u[3][6] = -1
            v[2][6] = 1; v[3][6] = 1
            w[1][6] = 1; w[3][6] = 1
            
            return u, v, w
        