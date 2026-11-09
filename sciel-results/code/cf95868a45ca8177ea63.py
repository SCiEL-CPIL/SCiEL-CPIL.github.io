            # Update c, b, fb
            c, fc = b, fb
            b, fb = x, fx
            
            if fa * fb > 0:
                a, fa = x, fx
