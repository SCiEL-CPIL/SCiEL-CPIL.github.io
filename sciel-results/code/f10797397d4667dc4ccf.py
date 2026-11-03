<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
=======
def _high_precision_polish(xy, r, deadline, max_iter=500):
    """
    Final high-precision polish with tighter tolerances.
    """
    n = len(xy)
    params = np.column_stack([xy, r])
    
    def obj(p):
        return _smooth_objective(p)
    
    def grad(p):
        return _smooth_grad(p)
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.49))
    
    result = minimize(obj, params, jac=grad, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
    
    out_params = result.x
    out_xy = out_params[:, :2]
    out_r = out_params[:, 2]
    
    for i in range(n):
        out_r[i] = max(0.001, min(0.49, out_r[i]))
        out_xy[i, 0] = np.clip(out_xy[i, 0], out_r[i], 1.0 - out_r[i])
        out_xy[i, 1] = np.clip(out_xy[i, 1], out_r[i], 1.0 - out_r[i])
    
    total, valid = _eval_packing(out_xy, out_r)
    if not valid:
        total = 0.0
    
    return out_xy, out_r, total, valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
>>>>>>> REPLACE

