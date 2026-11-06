import numpy as np

def construct():
    N = 8
    # Clenshaw-Curtis nodes on [-1, 1]
    # k = 0 to N-1
    k = np.arange(N)
    theta = k * np.pi / (N - 1)
    x = np.cos(theta)
    
    # Weights on [-1, 1]
    # w[0] = w[N-1] = 1/(N-1)
    # w[k] = 2/(N-1) * (1 - 2 * sum_{j odd} 1/j^2 * T_j(x_k))
    # T_j(x) = cos(j * arccos(x)) = cos(j * theta)
    
    weights = np.zeros(N)
    weights[0] = 1.0 / (N - 1)
    weights[-1] = 1.0 / (N - 1)
    
    for i in range(1, N - 1):
        s = 0.0
        for j in range(1, N - 1, 2): # j = 1, 3, ... N-2
            s += (1.0 / (j * j)) * np.cos(j * theta[i])
        weights[i] = (2.0 / (N - 1)) * (1.0 - 2.0 * s)
        
    # Map nodes to [0, 1]
    # x is decreasing from 1 to -1.
    # u = (x + 1) / 2
    # We want strictly increasing nodes.
    # x[0]=1 -> u=1
    # x[-1]=-1 -> u=0
    # So we reverse x to get increasing u.
    u_nodes = (x[::-1] + 1.0) / 2.0
    # Weights must correspond to the nodes.
    # x[0] had w[0]. x[-1] had w[-1].
    # Reversed x: x[-1] is first.
    # So weights should be reversed too.
    u_weights = weights[::-1]
    
    # Scale weights for [0, 1]
    # Integral_0^1 f(u) du = 0.5 * Integral_-1^1 f((v+1)/2) dv
    # So weights *= 0.5
    u_weights *= 0.5
    
    return {"nodes": u_nodes, "weights": u_weights}
