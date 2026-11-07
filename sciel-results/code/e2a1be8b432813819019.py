import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Nonlinear oscillator acceleration using a Gaussian RBF basis with fixed centers.
        
        Structural change: Replaces the sparse physical basis with a dense RBF basis.
        This allows the model to approximate a wider variety of nonlinear functions.
        The problem becomes linear in the parameters (weights), making the optimization
        landscape convex (a quadratic bowl), which BFGS can navigate efficiently.
        
        We use 9 Gaussian kernels centered on a 3x3 grid covering the input domain,
        plus 1 bias term.
        """
        import time
        start_time = time.time()
        
        # Unpack parameters
        # p0..p8 are RBF weights, p9 is bias
        w = params[:9]
        bias = params[9]
        
        # Define centers
        # x range approx [-0.6, 0.5], v range approx [-0.2, 0.3]
        x_centers = np.array([-0.4, 0.0, 0.4])
        v_centers = np.array([-0.15, 0.05, 0.25])
        
        # Create meshgrid of centers
        cx, cv = np.meshgrid(x_centers, v_centers, indexing='ij')
        cx = cx.flatten()
        cv = cv.flatten()
        
        # Calculate distances
        # x is (N,), v is (N,)
        # cx is (9,), cv is (9,)
        # We need dist^2 for each sample i and center j
        # dist2[i, j] = (x[i] - cx[j])^2 + (v[i] - cv[j])^2
        
        dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (N, 9)
        dv = v[:, np.newaxis] - cv[np.newaxis, :]  # (N, 9)
        dist2 = dx**2 + dv**2
        
        # Width sigma
        sigma = 0.25
        sigma2 = 2 * sigma**2
        
        # RBF values
        rbf = np.exp(-dist2 / sigma2)  # (N, 9)
        
        # Prediction
        pred = rbf @ w + bias
        
        # Time guard (though this is just a forward pass, it's fast)
        # The fitting is done by the evaluator, not here.
        # This function is called repeatedly during fitting.
        # It must be fast.
        
        return pred
