import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Use 9 RBFs + 1 constant term
    # Centers spread across the domain [0.1, 6]
    centers = np.linspace(0.2, 5.8, 9)
    # Uniform bandwidth for stability and local flexibility
    bandwidths = 0.4 * np.ones(9)
    
    rbf_values = np.zeros((len(x), 9))
    for i in range(9):
        rbf_values[:, i] = np.exp(-0.5 * ((x - centers[i]) / bandwidths[i])**2)
        
    const_term = np.ones((len(x), 1))
    
    basis = np.hstack((rbf_values, const_term))
    
    return basis @ params
