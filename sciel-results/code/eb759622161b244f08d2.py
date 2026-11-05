def strassen_2x2(i0, i1, j0, j1, k0, k1):
    nonlocal col
    ...
    add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1)
    add_term(i0*m + j1, j0*p + k1, k0*n + i0, 1) # Wait, U and V need to match the sum
    
    # Strassen steps:
    ...
    def add_strassen_term(u_entries, v_entries, w_entries):
        ...
    
    # 1. m1 = (a00+a01)*(b00+b01)
    add_strassen_term(...)
    ...
