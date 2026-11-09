def integrate(f, t_span, y0):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]
