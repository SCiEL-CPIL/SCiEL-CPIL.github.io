<<<<<<< SEARCH
def _optimize_generic(problem):
    """Optimize a generic unitary using gradient-based optimization."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Create a fixed CNOT template
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += ["u3" for _ in range(qubits)]
        layout.append(("cx", pairs[layer % len(pairs)]))
    layout += ["u3" for _ in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g == "u3")
    n_params = 3 * n_u3
    
    def build_from_params(params):
        """Build the unitary from parameter vector."""
        gates = []
        k = 0
        for gate in layout:
            if gate == "u3":
                theta, phi, lam = params[k], params[k+1], params[k+2]
                gates.append(["u3", 0, theta, phi, lam])  # placeholder, fix qubit
                k += 3
            else:
                gates.append(list(gate))
        # Fix qubit indices
        gates_fixed = []
        k = 0
        u3_idx = 0
        for gate in layout:
            if gate == "u3":
                theta, phi, lam = params[k], params[k+1], params[k+2]
                gates_fixed.append(["u3", u3_idx % qubits, theta, phi, lam])
                u3_idx += 1
                k += 3
            else:
                gates_fixed.append(list(gate))
        return gates_fixed
    
    def loss(params):
        """Compute the phase-insensitive distance."""
        gates = build_from_params(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Initialize parameters
    best_params = np.zeros(n_params)
    best_loss = loss(best_params)
    
    # Try multiple initializations
    n_starts = min(5, max(1, int(60 / max(1, n_params))))
    
    for start in range(n_starts):
        if time.time() > _DEADLINE - 10:
            break
        
        if start == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-math.pi, math.pi, size=n_params)
        
        try:
            result = minimize(loss, x0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Build final gate list
    gates = build_from_params(best_params)
    return gates
=======
def _nelder_mead(f, x0, max_iter=200, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Pure NumPy implementation of Nelder-Mead simplex optimization.
    f: objective function
    x0: initial guess
    """
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.1 if x0[i] != 0 else 0.1
        simplex.append(x_new)
    
    # Evaluate initial simplex
    for i in range(n + 1):
        simplex[i] = np.array([f(simplex[i])] + list(simplex[i]))
    
    # Sort simplex by objective function
    simplex = sorted(simplex, key=lambda x: x[0])
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Check convergence
        if simplex[-1][0] - simplex[0][0] < tol:
            break
            
        # Worst, second worst, best
        worst = simplex[-1]
        second_worst = simplex[-2]
        best = simplex[0]
        
        # Centroid of all points except the worst
        centroid = np.array([0.0] + [0.0] * n)
        for i in range(n):
            centroid = centroid + simplex[i]
        centroid = centroid / n
        
        # Reflection
        x_ref = centroid + alpha * (centroid - worst[1:])
        f_ref = f(x_ref)
        
        if f_ref < best[0]:
            # Expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = np.array([f_exp] + list(x_exp))
            else:
                simplex[-1] = np.array([f_ref] + list(x_ref))
        elif f_ref < second_worst[0]:
            simplex[-1] = np.array([f_ref] + list(x_ref))
        else:
            # Contraction
            x_con = centroid + rho * (worst[1:] - centroid)
            f_con = f(x_con)
            if f_con < worst[0]:
                simplex[-1] = np.array([f_con] + list(x_con))
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = np.array([f(simplex[0][1:] + sigma * (simplex[i][1:] - simplex[0][1:]))] + 
                                           list(simplex[0][1:] + sigma * (simplex[i][1:] - simplex[0][1:])))
        
        # Sort simplex
        simplex = sorted(simplex, key=lambda x: x[0])
        
    return simplex[0][1:]

def _optimize_generic(problem):
    """Optimize a generic unitary using Nelder-Mead optimization."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Create a fixed CNOT template
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += ["u3" for _ in range(qubits)]
        layout.append(("cx", pairs[layer % len(pairs)]))
    layout += ["u3" for _ in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g == "u3")
    n_params = 3 * n_u3
    
    def build_from_params(params):
        """Build the unitary from parameter vector."""
        gates_fixed = []
        k = 0
        u3_idx = 0
        for gate in layout:
            if gate == "u3":
                theta, phi, lam = params[k], params[k+1], params[k+2]
                gates_fixed.append(["u3", u3_idx % qubits, theta, phi, lam])
                u3_idx += 1
                k += 3
            else:
                gates_fixed.append(list(gate))
        return gates_fixed
    
    def loss(params):
        """Compute the phase-insensitive distance."""
        gates = build_from_params(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Initialize parameters
    best_params = np.zeros(n_params)
    best_loss = loss(best_params)
    
    # Try multiple initializations
    # Reduce starts if n_params is large to save time
    n_starts = min(3, max(1, int(30 / max(1, n_params))))
    
    for start in range(n_starts):
        if time.time() > _DEADLINE - 20:
            break
        
        if start == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-math.pi, math.pi, size=n_params)
        
        try:
            result = _nelder_mead(loss, x0, max_iter=100)
            res_loss = loss(result)
            if res_loss < best_loss:
                best_loss = res_loss
                best_params = result
        except Exception:
            pass
    
    # Build final gate list
    gates = build_from_params(best_params)
    return gates
>>>>>>> REPLACE
