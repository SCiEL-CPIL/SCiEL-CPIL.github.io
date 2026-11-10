import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Vector of distance from each node to the origin (node 0)
    dist_to_origin = distance_matrix[:, 0]
    
    # We want H[i, j] ~ prize[j] / (cost to go i->j and then j->0)
    # Cost is dist[i, j] + dist[j, 0]
    
    # Create a matrix where each column j is divided by (dist[i, j] + dist[j, 0])
    # dist[i, j] is the matrix itself.
    # dist[j, 0] is a vector, so we broadcast it to columns.
    
    denom = distance_matrix + dist_to_origin[None, :]
    
    # Avoid division by zero
    # If denom is 0, it implies dist[i,j]=0 and dist[j,0]=0.
    # This happens for i=j=0.
    # prize[0] is 0, so 0/0 is NaN.
    # We want to replace non-positive or non-finite values with 1e-9 later, 
    # but it's safer to handle here.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H = prizes[None, :] / denom
        
    # Replace NaN, Inf, or <= 0 with 1e-9
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # It also says "non-finite values ... make the whole result invalid" 
    # Wait, "An exception, non-finite values or a wrong shape make the whole result invalid."
    # This implies I should NOT return non-finite values.
    # So I must ensure the output contains no NaN or Inf.
    
    # np.where(H > 0, H, 1e-9) handles positive values.
    # But what about NaN? NaN > 0 is False. So it becomes 1e-9.
    # What about Inf? Inf > 0 is True. So it stays Inf.
    # We must check for finiteness.
    
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
