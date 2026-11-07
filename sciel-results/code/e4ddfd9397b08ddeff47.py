"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression via model selection from a portfolio of basis functions.
    Uses the training data (x, y) to select the best fitting model.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = len(x)

    if n == 0:
        return np.array([])

    # Build basis function portfolio
    def build_basis(x):
        """Return matrix of basis functions, one column per basis."""
        n = len(x)
        basis = [np.ones(n)]
        
        # Polynomials in x and y up to degree 3
        basis.append(x.copy())
        basis.append(y.copy())
        basis.append(x**2)
        basis.append(y**2)
        basis.append(x * y)
        basis.append(x**3)
        basis.append(y**3)
        basis.append(x**2 * y)
        basis.append(x * y**2)
        basis.append(x**3 * y)
        basis.append(x * y**3)
        
        # Trigonometric
        basis.append(np.sin(x))
        basis.append(np.sin(y))
        basis.append(np.cos(x))
        basis.append(np.cos(y))
        basis.append(np.sin(x * y))
        basis.append(np.cos(x * y))
        basis.append(np.sin(x + y))
        basis.append(np.sin(x - y))
        
        # Hyperbolic / activation-like
        basis.append(np.tanh(x))
        basis.append(np.tanh(y))
        basis.append(np.tanh(x * y))
        basis.append(np.tanh(x + y))
        
        # Exponential (clipped for stability)
        ex = np.exp(np.clip(x, -5, 5))
        ey = np.exp(np.clip(y, -5, 5))
        basis.append(ex)
        basis.append(ey)
        basis.append(ex * ey)
        
        # Rational-like (safe)
        basis.append(x / (1 + x**2))
        basis.append(y / (1 + y**2))
        basis.append(x * y / (1 + x**2 + y**2))
        
        # Log-like (safe)
        basis.append(np.sign(x) * np.log1p(np.abs(x)))
        basis.append(np.sign(y) * np.log1p(np.abs(y)))
        
        return np.column_stack(basis)
    
    Phi = build_basis(x)
    n_basis = Phi.shape[1]
    
    # Fit using ridge regression with different lambda values
    best_mse = np.inf
    best_pred = np.full(n, np.mean(y))
    
    # Try multiple regularization strengths
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    
    for lam in lambdas:
        try:
            if lam == 0.0:
                coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, y, rcond=None)
            else:
                # Ridge regression: (Phi^T Phi + lam*I)^{-1} Phi^T y
                A = Phi.T @ Phi + lam * np.eye(n_basis)
                b = Phi.T @ y
                try:
                    coeffs = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            
            pred = Phi @ coeffs
            if not np.all(np.isfinite(pred)):
                continue
            
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_pred = pred
        
        except Exception:
            continue
    
    # Also try forward selection with a subset of top features
    # Use correlation with y to rank features
    try:
        if n > 1:
            y_centered = y - np.mean(y)
            y_norm = np.sqrt(np.sum(y_centered ** 2)) + 1e-12
            feature_corrs = np.zeros(n_basis)
            for i in range(n_basis):
                f = Phi[:, i]
                f_centered = f - np.mean(f)
                f_norm = np.sqrt(np.sum(f_centered ** 2)) + 1e-12
                feature_corrs[i] = np.abs(np.sum(f_centered * y_centered)) / (f_norm * y_norm)
            
            # Sort by correlation
            order = np.argsort(feature_corrs)[::-1]
            
            # Try increasing subsets
            for k in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]:
                if time.time() > DEADLINE:
                    break
                if k > n_basis:
                    break
                idx = order[:k]
                Phi_sub = Phi[:, idx]
                
                for lam in [0.0, 1e-6, 1e-4, 1e-2]:
                    try:
                        if lam == 0.0:
                            coeffs, _, _, _ = np.linalg.lstsq(Phi_sub, y, rcond=None)
                        else:
                            A = Phi_sub.T @ Phi_sub + lam * np.eye(len(idx))
                            b = Phi_sub.T @ y
                            try:
                                coeffs = np.linalg.solve(A, b)
                            except np.linalg.LinAlgError:
                                coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
                        
                        pred = Phi_sub @ coeffs
                        if not np.all(np.isfinite(pred)):
                            continue
                        
                        mse = np.mean((pred - y) ** 2)
                        if mse < best_mse:
                            best_mse = mse
                            best_pred = pred
                    except Exception:
                        continue
    except Exception:
        pass
    
    # Final safety check
    if not np.all(np.isfinite(best_pred)):
        best_pred = np.full(n, np.mean(y))
    
    return best_pred
# EVOLVE-BLOCK-END
