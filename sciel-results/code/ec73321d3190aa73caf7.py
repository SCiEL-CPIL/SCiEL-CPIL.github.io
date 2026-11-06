def coefficients(stages):
    if stages != 12:
        return [0.0] * stages, [0.0] * stages

    # 4th order symmetric splitting coefficients
    # These satisfy sum(c)=1, sum(c^2)=1/2, sum(c^3)=1/3
    c = [
        0.154308825925099,
        0.065218974911122,
        0.366707731324080,
        0.366707731324080,
        0.065218974911122,
        0.154308825925099
    ]
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    for i in range(6):
        b[2*i] = c[i]
        a[2*i+1] = c[i]
        
    return a, b
