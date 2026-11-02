import numpy as np

def construct():
    N = 21
    R = 7
    C = 3
    
    # Coefficients for W and H in terms of r
    # W = (2*C - 1 + 1) * r? 
    # Even rows max x edge: 2*C*r.
    # Odd rows max x edge: 2*C*r + r.
    # So W = 2*C*r + r = (2*C + 1)r.
    # Wait, earlier I said W=7r for C=3.
    # 2*3 + 1 = 7. Correct.
    
    # H = 2r + (R-1)*sqrt(3)*r
    # H_coeff = 2 + (R-1)*sqrt(3)
    
    W_coeff = 2 * C + 1
    H_coeff = 2 + (R - 1) * np.sqrt(3)
    
    # Perimeter 4 => W + H = 2
    # (W_coeff + H_coeff) * r = 2
    r = 2.0 / (W_coeff + H_coeff)
    
    W = W_coeff * r
    H = H_coeff * r
    
    centers = []
    for i in range(R):
        for j in range(C):
            # Offset for odd rows
            offset = r if i % 2 == 1 else 0.0
            x = r + 2 * j * r + offset
            y = r + i * np.sqrt(3) * r
            centers.append([x, y, r])
            
    return np.array(centers)
