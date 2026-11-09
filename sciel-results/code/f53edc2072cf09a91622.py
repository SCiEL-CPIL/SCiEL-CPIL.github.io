<<<<<<< SEARCH
def _polish_decomposition(n, m, p, r, U, V, W, T, deadline, max_iters=200):
    """
    High-precision polish: use smaller learning rate and more iterations.
    """
    if time.time() > deadline:
        return U, V, W
    
    lr = 1e-5
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < 1e-12:
            break
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return U, V, W
=======
def _polish_decomposition(n, m, p, r, U, V, W, T, deadline, max_iters=500):
    """
    High-precision polish: use smaller learning rate and more iterations.
    """
    if time.time() > deadline:
        return U, V, W
    
    lr = 1e-4
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < 1e-14:
            break
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Decay learning rate
        if it % 100 == 99:
            lr *= 0.5
    
    return U, V, W
>>>>>>> REPLACE

