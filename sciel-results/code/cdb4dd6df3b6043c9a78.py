"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # UCB acquisition with cost-aware normalization.
    # Since we are minimizing, the "upper confidence bound" for the true value
    # is mu + beta * sigma. We want to maximize this to find potentially low values?
    # Wait, standard UCB for minimization is usually -mu + beta * sigma (maximizing negative)
    # or minimizing mu - beta * sigma.
    # Let's think carefully.
    # If we want to find the minimum, we are interested in points where the true value f(x) is low.
    # The GP predicts mu and sigma.
    # The "lower confidence bound" (LCB) for minimization is mu - beta * sigma.
    # We want to minimize LCB, which is equivalent to maximizing -LCB = -mu + beta * sigma.
    # Alternatively, we can define the acquisition as the potential improvement.
    #
    # The directive says: "Replace the standard Expected Improvement with a UCB (Upper Confidence Bound) acquisition function, UCB(x) = mu(x) + beta sigma(x)... divide this value by a linear cost penalty".
    # This phrasing is slightly tricky. Usually, UCB is used for maximization.
    # For minimization, the analogous bound that encourages exploration of low values is the Lower Confidence Bound (LCB) = mu - beta * sigma.
    # Maximizing (beta * sigma - mu) is equivalent to minimizing (mu - beta * sigma).
    #
    # Let's look at the standard EI for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # This is high when mu is low (exploitation) and sigma is high (exploration).
    #
    # The directive explicitly suggests: "UCB(x) = mu(x) + beta sigma(x)".
    # If we maximize mu + beta*sigma, we are looking for high mu? That seems wrong for minimization.
    # Unless the context implies that the "score" returned should be high for good points.
    # If we are minimizing, good points have low f(x).
    # Maybe the directive implies using the negative of the minimization objective?
    # Or perhaps it means the Upper Confidence Bound of the *regret*?
    #
    # Let's re-read carefully: "UCB(x) = mu(x) + beta sigma(x) ... divide this value by a linear cost penalty ... to balance exploitation, exploration, and evaluation cost".
    # If I use mu + beta*sigma, and I maximize it, I will pick points with high predicted mean. That is bad for minimization.
    # If I use -mu + beta*sigma, I am maximizing the "lower bound" on the negative value, or minimizing the "upper bound" on the positive value.
    # Actually, for minimization, the standard UCB-style acquisition is:
    # a(x) = -mu(x) + beta * sigma(x)
    # This is equivalent to minimizing mu(x) - beta * sigma(x).
    #
    # However, the directive *explicitly* writes the formula: $UCB(x) = \mu(x) + \beta \sigma(x)$.
    # Is it possible the task is framed as maximization? No, "minimizes hartmann 6d".
    # Is it possible the directive is slightly misstated and I should interpret "UCB" in the context of the *negative* objective?
    # If y = -f(x), we maximize y. Then UCB_y = mu_y + beta * sigma_y = -mu_f + beta * sigma_f.
    # This matches the logic for minimization.
    #
    # So, the acquisition score should be proportional to: (-mu + beta * sigma) / cost.
    #
    # Let's consider the cost term.
    # cost = 1 + mean(x). Since x is not passed, we don't know the exact cost per candidate.
    # Wait, the input `cost` has shape (2000,). This means the cost for each of the 2000 candidates is provided.
    # So we can divide by the specific cost for each candidate.
    #
    # Let's refine the formula:
    # score = (-mu + beta * sigma) / cost
    #
    # We need to choose beta. The directive suggests 2.0.
    # Let's try beta = 2.0.
    #
    # One issue: mu and sigma are in objective units.
    # Hartmann 6D minimum is around -3.3. f_best will be close to this.
    # The term -mu will be positive (since mu is negative).
    # sigma will be positive.
    # So the numerator is positive.
    # Dividing by cost (which is between 1 and 2) keeps it positive.
    # Higher score = better.
    #
    # Let's double check if "UCB" in the prompt might literally mean mu + beta*sigma and we are supposed to *minimize* the returned score?
    # "Return 2000 finite scores (higher = evaluate next)".
    # So we must maximize the return value.
    # If we return mu + beta*sigma, we maximize mu, which is bad.
    # Therefore, the intended formula for a minimization problem using a UCB-like approach must be effectively maximizing the potential to find a low value, i.e., maximizing (beta*sigma - mu).
    #
    # Let's implement: score = (beta * sigma - mu) / cost
    #
    # Is there any other interpretation?
    # Maybe "UCB" refers to the bound on the *regret*?
    # Regret r(x) = f(x) - f*.
    # We want small regret.
    # This is getting too abstract. The standard practice for minimization with GP is LCB (Lower Confidence Bound) which is mu - beta*sigma.
    # Minimizing LCB is equivalent to maximizing -LCB = -mu + beta*sigma.
    # This is exactly what I derived.
    #
    # Let's stick with: score = (beta * sigma - mu) / cost
    # with beta = 2.0.
    
    beta = 2.0
    
    # Ensure sigma is not zero to avoid division issues if we were doing something else, 
    # but here we just add it. 
    # However, if sigma is 0, the term is just -mu.
    
    # Calculate the UCB-style score for minimization
    # We want to maximize this score.
    # score = (beta * sigma - mu) / cost
    
    # Add a small epsilon to cost to avoid division by zero (though cost >= 1)
    scores = (beta * sigma - mu) / (cost + 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
