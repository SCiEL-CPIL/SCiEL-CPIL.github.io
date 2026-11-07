import numpy as np
import time

def acquisition(mu, sigma, f_best):
        start_time = time.time()
        
        # Thompson Sampling: Sample from the posterior
        # We want to minimize the function, so we prefer low values.
        # The evaluator maximizes the returned score.
        # So we return -sampled_values.
        
        # Ensure sigma is positive to avoid issues, though np.random.normal handles 0
        # sigma is standard deviation, so it should be >= 0.
        
        samples = np.random.normal(mu, sigma)
        
        # Safety check for deadline
        if time.time() - start_time > 295:
            # Fallback: return exploitation only (lowest mu)
            # This is a very fast operation, so unlikely to trigger
            return -mu
            
        return -samples
