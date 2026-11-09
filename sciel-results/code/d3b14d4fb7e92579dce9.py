# Rank 19 decomposition for 3x3 matrix multiplication
# U, V, W are 3x3 matrices
# C = sum_t W_t * (U_t . A) * (V_t . B) ? No.
# C_{ik} = sum_t W_t[i,k] * (sum_{j} U_t[i,j] A_{ij}) * (sum_{j} V_t[j,k] B_{jk})? No.
# The bilinear form is:
# C_{ik} = sum_t W_t[i,k] * alpha_t * beta_t
# where alpha_t = sum_{i',j'} U_t[i',j'] A_{i'j'}
# and beta_t = sum_{j',k'} V_t[j',k'] B_{j'k'}
# This requires U_t and V_t to be 3x3.
# The condition is sum_t W_t[i,k] U_t[i,j] V_t[j,k] = delta_{ij} delta_{jk} ? No.
# It must reproduce sum_j A_{ij} B_{jk}.
# So sum_t W_t[i,k] U_t[i,j] V_t[j,k] = 1 if j is the intermediate index?
# Actually, the tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
# So sum_t W_t[i,k] U_t[i,j] V_t[j,k] = delta_{ij} delta_{jk} is wrong.
# It should be: for fixed i, k, sum_j (sum_t W_t[i,k] U_t[i,j] V_t[j,k]) A_{ij} B_{jk} = sum_j A_{ij} B_{jk}.
# So we need sum_t W_t[i,k] U_t[i,j] V_t[j,k] = 1 for all i,j,k? No.
# The term A_{ij} B_{jk} appears in C_{ik}.
# So the coefficient of A_{ij} B_{jk} in C_{ik} must be 1.
# In the expansion sum_t W_t[i,k] (sum U_t A) (sum V_t B), the term A_{ij} B_{jk} has coefficient
# sum_t W_t[i,k] U_t[i,j] V_t[j,k].
# This must be 1 for all i,j,k.
# So we need sum_t W_t[i,k] U_t[i,j] V_t[j,k] = 1 for all i,j,k.
# This is the condition for the rank 19 decomposition.
