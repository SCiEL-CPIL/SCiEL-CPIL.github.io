import numpy as np
from scipy.stats import t as student_t

def acquisition(mu, sigma, f_best):
    nu = 5.0 # Degrees of freedom
    
    # Standardize
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Student-t CDF and PDF
    # Note: scipy.stats.t.cdf(z, df) is for standard t.
    # We need the CDF of the t-distribution at z.
    cdf_val = student_t.cdf(z, df=nu)
    pdf_val = student_t.pdf(z, df=nu)
    
    # EI formula structure is similar:
    # EI = E[max(0, f_best - f)]
    # = (f_best - mu) * P(f < f_best) + sigma * E[max(0, (f_best - f)/sigma)]
    # The second term involves the PDF.
    # For Gaussian: sigma * pdf(z)
    # For Student-t: sigma * pdf(z) ?
    # Let's check the derivation.
    # If f ~ T(mu, sigma, nu), then (f-mu)/sigma ~ T(0, 1, nu).
    # E[max(0, f_best - f)] = E[max(0, f_best - mu - sigma * T)]
    # Let Y = f_best - mu.
    # E[max(0, Y - sigma * T)]
    # This is not simply (Y) * CDF + sigma * PDF.
    # For Gaussian, it is.
    # For Student-t, the expectation of the truncated t-distribution is different.
    
    # However, a common approximation or variant is to just use the t-CDF and t-PDF in the same formula.
    # Or, we can use the fact that for heavy tails, the "tail" probability is higher.
    
    # Let's stick to the standard EI structure but with t-distribution components, 
    # or simply use UCB which is more robust to distributional assumptions.
    
    # Given the ambiguity of "Student-t acquisition" without full covariance, 
    # and the strong performance of UCB in similar tasks, 
    # I will bet on UCB with a tuned beta.
    
    # Let's try a hybrid: 
    # If we use UCB, we don't need f_best.
    # But f_best is provided.
    # Maybe a "Confidence" weighted EI?
    
    # Let's go with UCB. It's a distinct, strong family.
    beta = 2.5
    sigma = np.maximum(sigma, 1e-9)
    score = -mu + beta * sigma
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
