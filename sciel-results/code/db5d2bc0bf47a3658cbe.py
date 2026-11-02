import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Precompute constants
    normal_angles = np.array([np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def objective(params):
        xy = params[:2*N].reshape(N, 2)
        thetas = params[2*N:]
        
        projs = np.empty(N * 6 * 6)
        idx = 0
        for i in range(N):
            x, y = xy[i]
            theta = thetas[i]
            # Vertices
            for k in range(6):
                vx = x + np.cos(theta + k * np.pi / 3)
                vy = y + np.sin(theta + k * np.pi / 3)
                v = np.array([vx, vy])
                p = normals @ v
                projs[idx:idx+6] = p
                idx += 6
                
        # Smooth max
        alpha = 10.0
        max_p = np.max(projs)
        # Handle case where projs are very negative
        smooth_max = max_p + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (projs - max_p))))
        
        S = smooth_max / (np.sqrt(3) / 2)
        return S

    def constraints_func(params):
        xy = params[:2*N].reshape(N, 2)
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = xy[i][0] - xy[j][0]
                dy = xy[i][1] - xy[j][1]
                dist_sq = dx*dx + dy*dy
                cons.append(dist_sq - 3)
        return cons

    # Initial guesses
    guesses = []
    
    # Guess 1: Symmetric 7+4
    init_xy = [[0.0, 0.0]]
    for k in range(6):
        angle = np.pi / 6 + k * np.pi / 3
        init_xy.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    for k in range(4):
        angle = k * np.pi / 3
        init_xy.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
    guesses.append(np.concatenate([np.array(init_xy).flatten(), np.zeros(N)]))
    
    # Guess 2: Linear
    init_xy2 = [[i * np.sqrt(3), 0.0] for i in range(N)]
    guesses.append(np.concatenate([np.array(init_xy2).flatten(), np.zeros(N)]))
    
    # Guess 3: Random jitter
    rng = np.random.default_rng(42)
    base = np.concatenate([np.array(init_xy).flatten(), np.zeros(N)])
    jitter = rng.normal(0, 0.1, size=base.size)
    guesses.append(base + jitter)
    
    best_S = float('inf')
    best_params = None
    
    for guess in guesses:
        bounds = [(-10, 10)] * (2*N) + [(-np.pi, np.pi)] * N
        res = minimize(objective, guess, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': constraints_func}],
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and res.fun < best_S:
            best_S = res.fun
            best_params = res.x
            
    if best_params is None:
        # Fallback
        best_params = guesses[0]
        best_S = objective(best_params)
        
    xy = best_params[:2*N].reshape(N, 2)
    thetas = best_params[2*N:]
    thetas_deg = np.degrees(thetas)
    
    inner = []
    for i in range(N):
        inner.append([xy[i][0], xy[i][1], thetas_deg[i]])
        
    return inner, (0.0, 0.0), best_S, 0.0
