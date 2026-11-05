"""Broadband antireflection coating design via global + local optimization."""
import time
import numpy as np

def _reflectance_spectrum(thicknesses, n_layers, n_amb, n_sub, n_L, n_H, wavelengths):
    """
    Compute reflectance spectrum (in percent) for a given set of layer thicknesses.
    Uses the characteristic matrix method, vectorized over wavelengths.
    
    Parameters:
    -----------
    thicknesses : array-like, shape (n_layers,)
        Physical thicknesses in nm.
    n_layers : int
        Number of layers.
    n_amb : float
        Ambient index.
    n_sub : float
        Substrate index.
    n_L : float
        Low-index material index.
    n_H : float
        High-index material index.
    wavelengths : array-like, shape (N,)
        Wavelengths in nm.
    
    Returns:
    --------
    R : ndarray, shape (N,)
        Reflectance in percent at each wavelength.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    
    # Determine which layers are L and which are H
    # We'll encode this in the index array
    # For now, we need to know the sequence. Let's pass it or infer.
    # Actually, the design function returns both materials and thicknesses.
    # Let's make this function take the full stack definition.
    
    # This function is a helper; let's redefine it to take materials too.
    # For efficiency, let's just inline the logic in the main optimizer.
    pass

def design(problem):
    """
    Design a broadband antireflection coating.
    
    Parameters:
    -----------
    problem : dict
        Contains:
        - 'ambient_index': float
        - 'substrate_index': float
        - 'materials': dict with 'L' and 'H' keys
        - 'band_nm': list [lambda_min, lambda_max]
        - 'max_layers': int
        - 'max_total_nm': float
        - 'max_layer_nm': float
    
    Returns:
    --------
    materials : list of str
        List of 'L' or 'H' for each layer.
    thicknesses : list of float
        List of physical thicknesses in nm.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    n_L = problem['materials']['L']
    n_H = problem['materials']['H']
    lam_min, lam_max = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Wavelength grid for evaluation (401 points)
    wavelengths = np.linspace(lam_min, lam_max, 401)
    N_w = len(wavelengths)
    
    def compute_R(thicknesses, material_seq):
        """
        Compute mean reflectance (percent) for a given design.
        Vectorized over wavelengths.
        """
        n_layers = len(thicknesses)
        if n_layers == 0:
            # Bare substrate
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r * r
        
        # Build index array
        n_stack = np.zeros(n_layers)
        for i, mat in enumerate(material_seq):
            n_stack[i] = n_L if mat == 'L' else n_H
        
        # Phase terms: delta_i = 2*pi*n_i*d_i/lambda
        # thicknesses shape (n_layers,), wavelengths shape (N_w,)
        # We want delta shape (N_w, n_layers)
        # delta[j, i] = 2*pi*n_stack[i]*thicknesses[i]/wavelengths[j]
        
        # Vectorized: 
        # numerator: n_stack * thicknesses -> shape (n_layers,)
        # denominator: wavelengths -> shape (N_w,)
        # delta[j,i] = 2*pi * (n_stack[i]*thicknesses[i]) / wavelengths[j]
        
        nd = n_stack * thicknesses  # shape (n_layers,)
        # delta[j, i] = 2*pi*nd[i] / wavelengths[j]
        # Reshape for broadcasting: wavelengths[:, None] -> (N_w, 1), nd[None, :] -> (1, n_layers)
        delta = 2.0 * np.pi * nd[None, :] / wavelengths[:, None]  # shape (N_w, n_layers)
        
        # Characteristic matrices
        # For each layer i:
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        
        # We need to compute the product of all matrices, then combine with ambient and substrate.
        # This is tricky to vectorize directly because matrix multiplication is not element-wise.
        # However, for a small number of layers (<=20), we can loop over layers but vectorize over wavelengths.
        
        # Let's represent the characteristic matrix as a 2x2 complex array.
        # For each wavelength, we compute the product of matrices.
        # Since N_w=401 and n_layers<=20, we can do:
        # For each wavelength, multiply matrices sequentially.
        # But this is a Python loop over 401 wavelengths, each doing 20 matrix mults.
        # That's 8020 matrix mults, which is fast.
        
        # Alternative: use numpy's einsum or explicit formulas.
        # For efficiency, let's use the explicit 2x2 matrix multiplication.
        
        # Initialize M as identity for each wavelength
        M_real = np.ones((N_w, 2, 2))  # shape (N_w, 2, 2)
        M_imag = np.zeros((N_w, 2, 2))
        
        for i in range(n_layers):
            cos_d = np.cos(delta[:, i])  # shape (N_w,)
            sin_d = np.sin(delta[:, i])  # shape (N_w,)
            n_i = n_stack[i]
            
            # M_i = [[cos, i*sin/n_i], [i*n_i*sin, cos]]
            # M_i[0,0] = cos
            # M_i[0,1] = i*sin/n_i
            # M_i[1,0] = i*n_i*sin
            # M_i[1,1] = cos
            
            # Multiply M = M * M_i
            # M_new[0,0] = M[0,0]*M_i[0,0] + M[0,1]*M_i[1,0]
            # M_new[0,1] = M[0,0]*M_i[0,1] + M[0,1]*M_i[1,1]
            # M_new[1,0] = M[1,0]*M_i[0,0] + M[1,1]*M_i[1,0]
            # M_new[1,1] = M[1,0]*M_i[0,1] + M[1,1]*M_i[1,1]
            
            # Since matrices are complex, let's do it in complex form for clarity,
            # but for speed, we can separate real and imaginary parts.
            # Actually, let's just use complex arrays. It's simpler and likely fast enough.
            
            pass
        
        # Let's use complex arrays for simplicity
        M = np.ones((N_w, 2, 2), dtype=complex)
        
        for i in range(n_layers):
            cos_d = np.cos(delta[:, i])
            sin_d = np.sin(delta[:, i])
            n_i = n_stack[i]
            
            # M_i
            Mi = np.zeros((N_w, 2, 2), dtype=complex)
            Mi[:, 0, 0] = cos_d
            Mi[:, 0, 1] = 1j * sin_d / n_i
            Mi[:, 1, 0] = 1j * n_i * sin_d
            Mi[:, 1, 1] = cos_d
            
            # M = M @ Mi (matrix multiplication for each wavelength)
            # M_new[:,0,0] = M[:,0,0]*Mi[:,0,0] + M[:,0,1]*Mi[:,1,0]
            M_new = np.empty((N_w, 2, 2), dtype=complex)
            M_new[:, 0, 0] = M[:, 0, 0] * Mi[:, 0, 0] + M[:, 0, 1] * Mi[:, 1, 0]
            M_new[:, 0, 1] = M[:, 0, 0] * Mi[:, 0, 1] + M[:, 0, 1] * Mi[:, 1, 1]
            M_new[:, 1, 0] = M[:, 1, 0] * Mi[:, 0, 0] + M[:, 1, 1] * Mi[:, 1, 0]
            M_new[:, 1, 1] = M[:, 1, 0] * Mi[:, 0, 1] + M[:, 1, 1] * Mi[:, 1, 1]
            M = M_new
        
        # Admittance: Y = (M[1,0] + M[1,1]) / (M[0,0] + M[0,1])
        # But we need to include the substrate and ambient.
        # Actually, the standard formula is:
        # The total characteristic matrix is M_total = M_1 * M_2 * ... * M_N
        # Then the input admittance is Y_in = (M_total[1,0] + M_total[1,1]) / (M_total[0,0] + M_total[0,1])
        # But this is for the stack only. We need to include the substrate.
        
        # Correct approach:
        # The characteristic matrix of the stack (layers 1 to N) is M_stack.
        # The admittance at the air-side interface is:
        # Y_in = (M_stack[1,0] + n_sub * M_stack[1,1]) / (M_stack[0,0] + n_sub * M_stack[0,1])
        # Wait, I need to recall the exact formula.
        
        # From Hecht or Born & Wolf:
        # The characteristic matrix M relates the fields at the input and output planes.
        # For a stack on a substrate, the input admittance Y_in is:
        # Y_in = (B + C) / (A + D)  ... no, that's not right.
        
        # Let me use the standard formula:
        # If M = [[m11, m12], [m21, m22]] is the characteristic matrix of the stack,
        # and the substrate has index n_s, then the input admittance is:
        # Y_in = (m21 + n_s * m22) / (m11 + n_s * m12)
        
        # Then the reflectance is:
        # R = |(Y_in - n_amb) / (Y_in + n_amb)|^2
        
        m11 = M[:, 0, 0]
        m12 = M[:, 0, 1]
        m21 = M[:, 1, 0]
        m22 = M[:, 1, 1]
        
        Y_in = (m21 + n_sub * m22) / (m11 + n_sub * m12)
        
        R = np.abs((Y_in - n_amb) / (Y_in + n_amb)) ** 2
        return np.mean(R) * 100.0
    
    # Initial design: quarter-wave stack
    lam_center = 0.5 * (lam_min + lam_max)
    d_L = lam_center / (4.0 * n_L)
    d_H = lam_center / (4.0 * n_H)
    
    # Start with a few quarter-wave layers
    # Try different numbers of layers
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    # Try a simple 4-layer quarter-wave stack first
    init_materials = ['H', 'L', 'H', 'L']
    init_thicknesses = [d_H, d_L, d_H, d_L]
    
    # Check constraints
    if sum(init_thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in init_thicknesses):
        score = compute_R(init_thicknesses, init_materials)
        best_score = score
        best_materials = init_materials[:]
        best_thicknesses = init_thicknesses[:]
    
    # Now try to optimize using a simple local search
    # We'll use a random perturbation approach with annealing-like acceptance
    
    np.random.seed(42)
    
    current_materials = best_materials[:]
    current_thicknesses = np.array(best_thicknesses, dtype=float)
    current_score = best_score
    
    n_iter = 0
    max_iter = 50000  # Will be limited by deadline
    
    while time.time() < deadline and n_iter < max_iter:
        n_iter += 1
        
        # Random perturbation
        new_thicknesses = current_thicknesses.copy()
        new_materials = current_materials[:]
        
        # Perturb thicknesses
        for i in range(len(new_thicknesses)):
            if np.random.random() < 0.8:
                # Gaussian perturbation
                new_thicknesses[i] += np.random.normal(0, 20.0)
                new_thicknesses[i] = np.clip(new_thicknesses[i], 0, max_layer_nm)
        
        # Occasionally change material
        if np.random.random() < 0.1:
            idx = np.random.randint(len(new_materials))
            new_materials[idx] = 'H' if new_materials[idx] == 'L' else 'L'
        
        # Occasionally add or remove a layer
        if np.random.random() < 0.05 and len(new_materials) < max_layers:
            # Add a layer
            idx = np.random.randint(len(new_materials) + 1)
            new_mat = 'L' if np.random.random() < 0.5 else 'H'
            new_d = np.random.uniform(50, 200)
            new_materials.insert(idx, new_mat)
            new_thicknesses = np.insert(new_thicknesses, idx, new_d)
        elif np.random.random() < 0.05 and len(new_materials) > 1:
            # Remove a layer
            idx = np.random.randint(len(new_materials))
            new_materials.pop(idx)
            new_thicknesses = np.delete(new_thicknesses, idx)
        
        # Check constraints
        if sum(new_thicknesses) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in new_thicknesses):
            continue
        
        new_score = compute_R(new_thicknesses, new_materials)
        
        # Accept if better, or with some probability if worse (simulated annealing-like)
        if new_score < current_score:
            current_score = new_score
            current_thicknesses = new_thicknesses.copy()
            current_materials = new_materials[:]
            if new_score < best_score:
                best_score = new_score
                best_thicknesses = new_thicknesses.copy()
                best_materials = new_materials[:]
        else:
            # Accept with probability exp(-delta/T)
            T = max(0.01, 1.0 - n_iter / max_iter)
            if np.random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                current_thicknesses = new_thicknesses.copy()
                current_materials = new_materials[:]
    
    return best_materials, list(best_thicknesses)
