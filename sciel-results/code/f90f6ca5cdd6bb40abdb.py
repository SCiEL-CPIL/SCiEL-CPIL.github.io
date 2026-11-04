    # M7
    U[ai*m + (aj+1)][col] += 1
    U[(ai+1)*m + (aj+1)][col] -= 1
    V[(aj+1)*p + bk][col] += 1
    V[(aj+1)*p + (bk+1)][col] += 1
    W[bk*n + ai][col] += 1
    W[(bk+1)*n + (ai+1)][col] -= 1  # ERROR: Should not touch C11? 
    