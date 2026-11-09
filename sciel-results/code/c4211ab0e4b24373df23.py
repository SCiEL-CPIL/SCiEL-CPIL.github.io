import numpy as np

def integrate(f, t_span, y0):
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        t = t0
        
        # Fixed step size
        n_steps = 5
        h = (t1 - t0) / n_steps
        
        # Taylor order
        M = 50
        
        for _ in range(n_steps):
            # 1. Call f to get y'
            d1 = f(t, y)
            
            # 2. Compute higher derivatives analytically
            # d2 = J d1
            d2 = np.array([d1[1], -d1[0]])
            # d3 = J d2 = -d1
            d3 = -d1
            # d4 = J d3 = -J d1 = -d2
            d4 = -d2
            
            # Derivatives cycle every 4
            # d_{k} depends on k % 4
            # We sum h^k/k! * d_k
            # k=1: h/1! * d1
            # k=2: h^2/2! * d2
            # k=3: h^3/3! * d3
            # k=4: h^4/4! * d4
            # ...
            
            y_new = y.copy()
            term = h # h^1 / 1!
            # Add k=1
            y_new += term * d1
            
            # We can loop k from 2 to M
            # term_k = term_{k-1} * h / k
            # d_k based on k % 4
            # k=2: d2, k=3: d3, k=4: d4, k=5: d1...
            
            current_term = h
            for k in range(2, M + 1):
                current_term *= h / k
                if k % 4 == 2:
                    y_new += current_term * d2
                elif k % 4 == 3:
                    y_new += current_term * d3
                elif k % 4 == 0:
                    y_new += current_term * d4
                else: # k % 4 == 1
                    y_new += current_term * d1
            
            y = y_new
            t += h
            
        return y
