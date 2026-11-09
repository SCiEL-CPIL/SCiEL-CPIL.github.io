import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
       x = np.asarray(x, dtype=np.float64)
       p = np.asarray(params, dtype=np.float64)
       
       # Ensure p has exactly 10 elements
       if p.size < 10:
           p = np.concatenate([p, np.zeros(10 - p.size)])
       p = p[:10]
       
       # 2 Chebyshev terms
       t = x / 2.0
       T0 = np.ones_like(x)
       T1 = t
       
       # 8 RBF terms
       centers = np.linspace(-1.8, 1.8, 8)
       widths = 0.35
       
       rbf_terms = []
       for c in centers:
           rbf = np.exp(-((x - c)**2) / (2.0 * widths**2))
           rbf_terms.append(rbf)
       
       # Combine
       y_hat = (p[0] * T0 + 
                p[1] * T1 + 
                p[2] * rbf_terms[0] + 
                p[3] * rbf_terms[1] + 
                p[4] * rbf_terms[2] + 
                p[5] * rbf_terms[3] + 
                p[6] * rbf_terms[4] + 
                p[7] * rbf_terms[5] + 
                p[8] * rbf_terms[6] + 
                p[9] * rbf_terms[7])
                
       return y_hat
