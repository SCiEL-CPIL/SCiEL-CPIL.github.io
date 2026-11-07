import numpy as np

def acquisition(mu, sigma, f_best, cost):
            # Structural Change: Use Cost-Penalized UCB instead of EI/Cost
            # UCB encourages exploration of uncertain regions, which is beneficial
            # for multimodal functions like Ackley in low-budget settings.
            
            kappa = 2.0
            lambda_cost = 0.5
            
            # UCB for minimization: we want to maximize -mu + kappa * sigma
            ucb = -mu + kappa * sigma
            
            # Penalize cost
            scores = ucb - lambda_cost * cost
            
            # Ensure finite output
            scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
            
            return scores
