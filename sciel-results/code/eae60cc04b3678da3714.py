    def compute_residual(U_curr, V_curr, W_curr):
        """Compute residual and gradients efficiently using vectorized ops."""
        grad_U = np.zeros_like(U_curr)
        grad_V = np.zeros_like(V_curr)
        grad_W = np.zeros_like(W_curr)
        residual_sum = 0.0
        
        # Vectorize over valid triples
        for (a, b, c) in valid_triples:
            val = np.dot(U_curr[a, :], np.dot(V_curr[b, :], W_curr[c, :]))
            err = val - 1.0
            residual_sum += err * err
            
            # Gradient computation: 
            # d/dU[a,r] (sum_r' U[a,r'] V[b,r'] W[c,r']) = V[b,r] W[c,r]
            # d/dV[b,r] ... = U[a,r] W[c,r]
            # d/dW[c,r] ... = U[a,r] V[b,r]
            
            grad_U[a, :] += err * (V_curr[b, :] * W_curr[c, :])
            grad_V[b, :] += err * (U_curr[a, :] * W_curr[c, :])
            grad_W[c, :] += err * (U_curr[a, :] * V_curr[b, :])
        
        return residual_sum, grad_U, grad_V, grad_W
