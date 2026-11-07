import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import random

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin

    n_samples, n_features = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    
    if y_std < 1e-10:
        def predict_const(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
        return predict_const

    # Normalize X for stability in GP
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std

    y_norm = (y_train - y_mean) / y_std

    # --- Gaussian Process Baseline (Current Best) ---
    # We keep the GP as a strong baseline.
    def matern52_kernel(X1, X2, lengthscales, variance):
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * np.dot(X1_scaled, X2_scaled.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        dists = np.sqrt(sq_dists)
        sqrt5 = np.sqrt(5)
        K = variance * (1 + sqrt5 * dists + (5/3) * dists**2) * np.exp(-sqrt5 * dists)
        return K

    def log_marginal_likelihood(theta):
        if time.time() > deadline - 10:
            return np.inf
        log_l = theta[:n_features]
        log_sigma_f = theta[n_features]
        log_sigma_n = theta[n_features + 1]
        lengthscales = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        K = matern52_kernel(X_norm, X_norm, lengthscales, sigma_f**2)
        K[np.diag_indices_from(K)] += sigma_n**2
        try:
            L, lower = cho_factor(K, lower=True, check_finite=False)
            alpha, _ = cho_solve((L, lower), y_norm, check_finite=False)
            logdet = 2 * np.sum(np.log(np.diag(lower)))
            lml = -0.5 * np.dot(y_norm, alpha) - 0.5 * logdet - 0.5 * n_samples * np.log(2 * np.pi)
            if not np.isfinite(lml):
                return np.inf
            return -lml
        except Exception:
            return np.inf

    # Initialize GP
    init_l = np.zeros(n_features)
    x0_gp = np.concatenate([init_l, [np.log(1.0), np.log(0.01)]])
    bounds_gp = [(-5, 5)] * (n_features + 2)
    
    best_gp_mse = np.inf
    best_gp_predict = None

    try:
        res_gp = minimize(log_marginal_likelihood, x0_gp, method='L-BFGS-B', bounds=bounds_gp, 
                       options={'maxiter': 20, 'maxfun': 50, 'ftol': 1e-6})
        
        final_log_l = res_gp.x[:n_features]
        final_log_sigma_f = res_gp.x[n_features]
        final_log_sigma_n = res_gp.x[n_features + 1]
        final_lengthscales = np.exp(final_log_l)
        final_sigma_f = np.exp(final_log_sigma_f)
        final_sigma_n = np.exp(final_log_sigma_n)
        
        K_train = matern52_kernel(X_norm, X_norm, final_lengthscales, final_sigma_f**2)
        K_train[np.diag_indices_from(K_train)] += final_sigma_n**2
        L_train, lower_train = cho_factor(K_train, lower=True, check_finite=False)
        alpha, _ = cho_solve((L_train, lower_train), y_norm, check_finite=False)
        
        # Evaluate GP MSE on training data to compare
        K_test_train = matern52_kernel(X_norm, X_norm, final_lengthscales, final_sigma_f**2)
        pred_norm = K_test_train @ alpha
        mse_gp = np.mean((pred_norm - y_norm)**2)
        best_gp_mse = mse_gp
        
        def predict_gp(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_n = (X - X_mean) / X_std
            K_test = matern52_kernel(X_n, X_norm, final_lengthscales, final_sigma_f**2)
            mean_norm = K_test @ alpha
            mean_orig = mean_norm * y_std + y_mean
            return mean_orig

        best_gp_predict = predict_gp
    except Exception:
        best_gp_mse = np.inf
        best_gp_predict = None

    # --- Genetic Programming Search ---
    # We will search for a symbolic expression.
    # Primitives: +, -, *, /, sin, cos, exp, log, abs, sqrt
    # Terminals: x1, x2, x3, 1
    
    class Node:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children or []
        
        def evaluate(self, X):
            if self.name in ['x1', 'x2', 'x3']:
                idx = int(self.name[1]) - 1
                return X[:, idx]
            elif self.name == '1':
                return np.ones(X.shape[0])
            elif self.name == '+':
                return self.children[0].evaluate(X) + self.children[1].evaluate(X)
            elif self.name == '-':
                return self.children[0].evaluate(X) - self.children[1].evaluate(X)
            elif self.name == '*':
                return self.children[0].evaluate(X) * self.children[1].evaluate(X)
            elif self.name == '/':
                den = self.children[1].evaluate(X)
                den[abs(den) < 1e-8] = 1e-8
                return self.children[0].evaluate(X) / den
            elif self.name == 'sin':
                return np.sin(self.children[0].evaluate(X))
            elif self.name == 'cos':
                return np.cos(self.children[0].evaluate(X))
            elif self.name == 'exp':
                val = self.children[0].evaluate(X)
                val = np.clip(val, -5, 5)
                return np.exp(val)
            elif self.name == 'log':
                val = self.children[0].evaluate(X)
                val = np.clip(val, 1e-8, 100)
                return np.log(val)
            elif self.name == 'abs':
                return np.abs(self.children[0].evaluate(X))
            elif self.name == 'sqrt':
                val = self.children[0].evaluate(X)
                val = np.clip(val, 0, 100)
                return np.sqrt(val)
            else:
                return np.zeros(X.shape[0])
        
        def depth(self):
            if not self.children:
                return 0
            return 1 + max(c.depth() for c in self.children)
        
        def copy(self):
            children = [c.copy() for c in self.children]
            return Node(self.name, children)
        
        def __repr__(self):
            if not self.children:
                return self.name
            return f"{self.name}({', '.join(repr(c) for c in self.children)})"

    def random_tree(depth=0, max_depth=4):
        if depth >= max_depth:
            return Node(random.choice(['x1', 'x2', 'x3', '1'][:n_features+1] if n_features < 3 else ['x1', 'x2', 'x3', '1']))
        
        if random.random() < 0.3:
            return Node(random.choice(['x1', 'x2', 'x3', '1'][:n_features+1] if n_features < 3 else ['x1', 'x2', 'x3', '1']))
        
        if random.random() < 0.5:
            func = random.choice(['+', '-', '*', '/'])
            return Node(func, [random_tree(depth+1, max_depth), random_tree(depth+1, max_depth)])
        else:
            func = random.choice(['sin', 'cos', 'exp', 'log', 'abs', 'sqrt'])
            return Node(func, [random_tree(depth+1, max_depth)])

    def mutate(tree, depth=0, max_depth=4):
        if random.random() < 0.3:
            return random_tree(depth, max_depth)
        children = [mutate(c, depth+1, max_depth) for c in tree.children]
        return Node(tree.name, children)

    def crossover(tree1, tree2, depth=0, max_depth=4):
        if random.random() < 0.5:
            return tree1.copy()
        if random.random() < 0.5:
            return tree2.copy()
        children = [crossover(c, tree2, depth+1, max_depth) if random.random() < 0.5 else crossover(c, tree1, depth+1, max_depth) for c in tree1.children]
        return Node(tree1.name, children)

    best_tree = None
    best_tree_mse = np.inf
    best_w = 1.0
    best_b = 0.0

    population = [random_tree(0, 3) for _ in range(50)]
    
    for gen in range(20):
        if time.time() > deadline - 20:
            break
            
        scored_pop = []
        for tree in population:
            try:
                vals = tree.evaluate(X_norm)
                if not np.all(np.isfinite(vals)):
                    continue
                # Fit y = w * vals + b
                A = np.column_stack([vals, np.ones(n_samples)])
                w, _, _, _ = np.linalg.lstsq(A, y_norm, rcond=None)
                pred = w[0] * vals + w[1]
                mse = np.mean((pred - y_norm)**2)
                if mse < best_tree_mse:
                    best_tree_mse = mse
                    best_tree = tree
                    best_w = w[0]
                    best_b = w[1]
                scored_pop.append((mse, tree))
            except Exception:
                continue
        
        if not scored_pop:
            break
            
        scored_pop.sort(key=lambda x: x[0])
        
        # Selection
        new_pop = []
        # Keep top 10
        for i in range(10):
            new_pop.append(scored_pop[i][1].copy())
            
        # Generate new
        while len(new_pop) < 50:
            if random.random() < 0.5:
                t1, t2 = random.sample(scored_pop[:20], 2)
                new_pop.append(crossover(t1[1], t2[1], 0, 4))
            else:
                t1 = random.choice(scored_pop[:20])
                new_pop.append(mutate(t1[1], 0, 4))
        
        population = new_pop

    # --- Final Selection ---
    # Compare GP (Gaussian) and GP (Genetic)
    # We need to calculate the actual MSE for the Genetic GP on the training set
    # and compare it with the Gaussian GP MSE.
    
    # Note: best_tree_mse is in normalized y space.
    # best_gp_mse is in normalized y space.
    
    if best_tree is not None and best_tree_mse < best_gp_mse:
        # Use Genetic GP
        final_tree = best_tree
        final_w = best_w
        final_b = best_b
        
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_n = (X - X_mean) / X_std
            vals = final_tree.evaluate(X_n)
            pred_norm = final_w * vals + final_b
            pred_orig = pred_norm * y_std + y_mean
            return pred_orig
        return predict
    else:
        # Use Gaussian GP
        if best_gp_predict is not None:
            return best_gp_predict
        else:
            # Fallback: Linear Regression
            X_design = np.hstack([X_train, np.ones((n_samples, 1))])
            w, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
            def predict_lin(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                X_pred = np.hstack([X, np.ones((X.shape[0], 1))])
                return X_pred @ w
            return predict_lin
