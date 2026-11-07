"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with quadratic surrogate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        history_x = [x.copy()]
        history_f = [fx]
        max_history = 25

        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Try quadratic surrogate if we have enough points
            if len(history_x) >= 8:
                try:
                    X = np.array(history_x[-max_history:])
                    F = np.array(history_f[-max_history:])
                    
                    # Fit quadratic model: f(x) ≈ c + b^T x + x^T H x
                    # Use centering for numerical stability
                    x_center = X.mean(axis=0)
                    Xc = X - x_center
                    Fc = F - F.mean()
                    
                    # Build feature matrix for quadratic terms
                    n = len(Xc)
                    d = self.dim
                    # Features: [1, x_i, x_i*x_j for i<=j]
                    n_feat = 1 + d + d * (d + 1) // 2
                    Phi = np.zeros((n, n_feat))
                    Phi[:, 0] = 1
                    Phi[:, 1:1+d] = Xc
                    idx = 1 + d
                    for i in range(d):
                        for j in range(i, d):
                            Phi[:, idx] = Xc[:, i] * Xc[:, j]
                            idx += 1
                    
                    # Solve least squares
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(Phi, Fc, rcond=None)
                    except:
                        coeffs = None
                    
                    if coeffs is not None:
                        # Extract Hessian and gradient from coefficients
                        # coeffs[0] = constant, coeffs[1:1+d] = linear, rest = quadratic
                        b = coeffs[1:1+d]
                        H = np.zeros((d, d))
                        idx = 1 + d
                        for i in range(d):
                            for j in range(i, d):
                                H[i, j] = coeffs[idx]
                                H[j, i] = coeffs[idx]
                                idx += 1
                        
                        # Add regularization for stability
                        H_reg = H + 1e-6 * np.eye(d)
                        
                        # Check if H is positive definite (local minimum)
                        try:
                            eigvals = np.linalg.eigvalsh(H_reg)
                            if np.all(eigvals > 1e-8):
                                # Solve H * dx = -b for the step
                                dx = np.linalg.solve(H_reg, -b)
                                # Clip step to reasonable size
                                step_norm = np.linalg.norm(dx)
                                if step_norm > 5.0:
                                    dx = dx / step_norm * 5.0
                                elif step_norm < 1e-6:
                                    dx = 0.0
                                
                                y = np.clip(x + dx, -5, 5)
                                fy = func(y)
                                history_x.append(y.copy())
                                history_f.append(fy)
                                if len(history_x) > max_history:
                                    history_x.pop(0)
                                    history_f.pop(0)
                                if fy < fx:
                                    x, fx = y, fy
                                    # Reduce sigma since we're making progress
                                    sigma *= 0.9
                                else:
                                    sigma *= 0.85
                                continue
                        except:
                            pass
                except:
                    pass

            # Fallback: standard (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            history_x.append(y.copy())
            history_f.append(fy)
            if len(history_x) > max_history:
                history_x.pop(0)
                history_f.pop(0)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        return x, fx
# EVOLVE-BLOCK-END
