def _construct_rank16_valid(n=2, m=2, p=5):
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen for cols 0,1 (terms 0-6)
    # ... (same as before)
    
    # For cols 2,3,4 (terms 7-15): 9 terms
    # Need to cover:
    # C02 = A00*B02 + A01*B12
    # C03 = A00*B03 + A01*B13
    # C04 = A00*B04 + A01*B14
    # C12 = A10*B02 + A11*B12
    # C13 = A10*B03 + A11*B13
    # C14 = A10*B04 + A11*B14
