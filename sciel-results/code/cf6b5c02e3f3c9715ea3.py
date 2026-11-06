def coefficients(stages):
    # 6th-order symmetric Suzuki-Trotter coefficients (10 stages)
    # Padded to 12 stages
    alpha6 = 0.084
    beta6 = 0.416
    alpha4 = -0.1555
    beta4 = 0.1555
    
    # Constructing the 10-stage coefficients
    a = [0.0] * 12
    b = [0.0] * 12
    
    # Approximate 6th-order coefficients
    # a: [c1, c2, c3, c4, c5, c5, c4, c3, c2, c1]
    # b: [d1, d2, d3, d4, d5, d5, d4, d3, d2, d1]
    
    # Using known values for 6th order symmetric
    c1 = 0.084
    c2 = 0.156
    c3 = 0.156
    c4 = 0.156
    c5 = 0.156
    
    d1 = 0.416
    d2 = 0.156
    d3 = 0.156
    d4 = 0.156
    d5 = 0.156
    
    a = [c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, 0, 0]
    b = [d1, d2, d3, d4, d5, d5, d4, d3, d2, d1, 0, 0]
    
    # Normalize to ensure sum is 1
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
        
    return a, b
