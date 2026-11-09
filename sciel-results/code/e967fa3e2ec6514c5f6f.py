import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Optimize p for 2-point distribution on {0, M}
    # Maximize P(p) = p(1-p)^2(1+2p)
    # Derivative root: 8p^3 - 9p^2 + 1 = 0 => (p-1)^2(8p+1)=0
    # Optimal p in (0,1) is (1 + sqrt(33)) / 16
    p = (1.0 + np.sqrt(33.0)) / 16.0
    support = np.array([0.0, 1000.0], dtype=np.float64)
    weights = np.array([1.0 - p, p], dtype=np.float64)
    return support, weights
