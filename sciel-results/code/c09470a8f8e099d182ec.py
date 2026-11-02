import numpy as np
import time

N = 400

def construct():
        start_time = time.time()
        deadline = start_time + 110  # 10s margin
        
        # Initialize
        x = np.random.choice([-1, 1], size=N)
        
        # Helper to compute max sidelobe
        def get_max_sidelobe(x):
            # Autocorrelation
            # Using np.correlate
            ac = np.correlate(x, x, mode='full')
            # Peak is at N-1
            peak = ac[N-1]
            # Sidelobes are the rest
            # ac[0:N-1] and ac[N+1:2N]
            # We want max abs of these
            # Note: ac is symmetric
            # Just take ac[0:N-1]
            if N == 1: return 0
            sidelobes = np.abs(ac[:N-1])
            return np.max(sidelobes)

        current_max = get_max_sidelobe(x)
        best_max = current_max
        best_x = x.copy()
        
        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        
        while time.time() < deadline:
            # Pick random index
            i = np.random.randint(N)
            
            # Flip
            x[i] *= -1
            
            # Evaluate
            new_max = get_max_sidelobe(x)
            delta = new_max - current_max
            
            if delta < 0:
                # Accept
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
            else:
                # Probabilistic accept
                if T > min_T:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_max = new_max
                    else:
                        # Revert
                        x[i] *= -1
                else:
                    # Revert
                    x[i] *= -1
            
            # Cool
            T *= cooling_rate
            
        return best_x
