import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1]
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 5]
    xs = (s - 1.0) / 4.0
    # T: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Basis functions
    # 1. Constant
    phi1 = np.ones_like(xb)
    # 2. Linear b
    phi2 = xb
    # 3. Linear s
    phi3 = xs
    # 4. Linear T
    phi4 = xT
    # 5. Linear pH
    phi5 = xPH
    # 6. Quadratic b
    phi6 = xb**2
    # 7. Quadratic s
    phi7 = xs**2
    # 8. Quadratic T
    phi8 = xT**2
    # 9. Quadratic pH
    phi9 = xPH**2
    # 10. Interaction T-pH
    phi10 = xT * xPH
    
    # Linear combination
    return (p[0] * phi1 + p[1] * phi2 + p[2] * phi3 + p[3] * phi4 + p[4] * phi5 +
            p[5] * phi6 + p[6] * phi7 + p[7] * phi8 + p[8] * phi9 + p[9] * phi10)
