def strassen(ac0, ac1, ar0, ar1, bc0, bc1, br0, br1, cc0, cc1, cr0, cr1):
    # P1 = (A[ar0,ac0]+A[ar1,ac1])(B[br0,bc0]+B[br1,bc1])
    add_term([((ar0, ac0), 1), ((ar1, ac1), 1)],
             [((br0, bc0), 1), ((br1, bc1), 1)],
             [((cr0, cc0), 1), ((cr1, cc1), 1)])
    # P2 = (A[ar1,ac0]+A[ar1,ac1])B[br0,bc0]
    add_term([((ar1, ac0), 1), ((ar1, ac1), 1)],
             [((br0, bc0), 1)],
             [((cr1, cc0), 1), ((cr1, cc1), -1)])
    # P3 = A[ar0,ac0](B[br0,bc1]-B[br1,bc1])
    add_term([((ar0, ac0), 1)],
             [((br0, bc1), 1), ((br1, bc1), -1)],
             [((cr0, cc1), 1), ((cr1, cc1), 1)])
    # P4 = A[ar1,ac1](B[br1,bc0]-B[br0,bc0])
    add_term([((ar1, ac1), 1)],
             [((br1, bc0), 1), ((br0, bc0), -1)],
             [((cr0, cc0), 1), ((cr1, cc0), 1)])
    # P5 = (A[ar0,ac1]-A[ar1,ac1])B[br1,bc1]
    add_term([((ar0, ac1), 1), ((ar1, ac1), -1)],
             [((br1, bc1), 1)],
             [((cr0, cc0), -1), ((cr0, cc1), 1)])
    # P6 = (A[ar1,ac0]-A[ar0,ac0])(B[br0,bc0]+B[br0,bc1])
    add_term([((ar1, ac0), 1), ((ar0, ac0), -1)],
             [((br0, bc0), 1), ((br0, bc1), 1)],
             [((cr1, cc1), 1)])
    # P7 = (A[ar0,ac1]-A[ar1,ac1])(B[br1,bc0]+B[br1,bc1])
    add_term([((ar0, ac1), 1), ((ar1, ac1), -1)],
             [((br1, bc0), 1), ((br1, bc1), 1)],
             [((cr0, cc0), 1)])
