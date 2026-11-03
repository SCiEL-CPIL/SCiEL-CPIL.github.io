<<<<<<< SEARCH
    try:
        result = minimize(lambda x: x[2], np.array([0,0,1]), 
                          constraints=[{'type': 'ineq', 'fun': lambda x, k=k: Q_vals[k] - (x[0]*u_vectors[k][0] + x[1]*u_vectors[k][1] - x[2]) for k in range(6)},
                                       {'type': 'ineq', 'fun': lambda x, k=k: -P_vals[k] - (-x[0]*u_vectors[k][0] - x[1]*u_vectors[k][1] - x[2]) for k in range(6)}],
                          method='SLSQP')
        # This SLSQP might be slow or fail. Let's use linprog from scipy.
        from scipy.optimize import linprog
=======
    try:
        from scipy.optimize import linprog
>>>>>>> REPLACE
