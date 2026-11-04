def strassen_block(i0, j0, k0):
    # i0, j0, k0 are top-left indices
    # Local 0,1 map to global i0, i0+1 etc.
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    u = {(i0+0)*m + (j0+0): 1, (i0+1)*m + (j0+1): 1}
    v = {(j0+0)*p + (k0+0): 1, (j0+1)*p + (k0+1): 1}
    w = {(k0+0)*n + (i0+0): 1, (k0+1)*n + (i0+1): 1}
    add_term_from_dicts(u, v, w)

    # M2: (A10+A11)B00 -> C10+1
    u = {(i0+1)*m + (j0+0): 1, (i0+1)*m + (j0+1): 1}
    v = {(j0+0)*p + (k0+0): 1}
    w = {(k0+0)*n + (i0+1): 1}
    add_term_from_dicts(u, v, w)

    # M3: A00(B10-B00) -> C01+1
    u = {(i0+0)*m + (j0+0): 1}
    v = {(j0+1)*p + (k0+0): 1, (j0+0)*p + (k0+0): -1}
    w = {(k0+1)*n + (i0+0): 1}
    add_term_from_dicts(u, v, w)

    # M4: A11(B01-B00) -> C00+1, C11+1
    u = {(i0+1)*m + (j0+1): 1}
    v = {(j0+0)*p + (k0+1): 1, (j0+0)*p + (k0+0): -1}
    w = {(k0+0)*n + (i0+0): 1, (k0+1)*n + (i0+1): 1}
    add_term_from_dicts(u, v, w)

    # M5: (A00+A01)B11 -> C00-1, C01+1
    u = {(i0+0)*m + (j0+0): 1, (i0+0)*m + (j0+1): 1}
    v = {(j0+1)*p + (k0+1): 1}
    w = {(k0+0)*n + (i0+0): -1, (k0+1)*n + (i0+0): 1}
    add_term_from_dicts(u, v, w)

    # M6: (A10-A00)(B00+B01) -> C11+1
    u = {(i0+1)*m + (j0+0): 1, (i0+0)*m + (j0+0): -1}
    v = {(j0+0)*p + (k0+0): 1, (j0+0)*p + (k0+1): 1}
    w = {(k0+1)*n + (i0+1): 1}
    add_term_from_dicts(u, v, w)

    # M7: (A01-A11)(B10+B11) -> C00+1
    u = {(i0+0)*m + (j0+1): 1, (i0+1)*m + (j0+1): -1}
    v = {(j0+1)*p + (k0+0): 1, (j0+1)*p + (k0+1): 1}
    w = {(k0+0)*n + (i0+0): 1}
    add_term_from_dicts(u, v, w)
