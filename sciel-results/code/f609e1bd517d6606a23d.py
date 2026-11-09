import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
       x = np.asarray(x, dtype=np.float64)
       y = np.asarray(y, dtype=np.float64)
       z = np.asarray(z, dtype=np.float64)
       p = np.asarray(params, dtype=np.float64)
       
       # Model: p0 * x^p1 * y^p2 * z^p3 + p4 * x^p5 * y^p6 * z^p7 + p8 * (x+y+z) + p9
       
       # Calculate Term 1
       # Using log to avoid intermediate overflow?
       # log_term1 = p1*log(x) + p2*log(y) + p3*log(z)
       # term1 = p0 * exp(log_term1)
       # This is safer if p1..p3 are large.
       
       log_x = np.log(x)
       log_y = np.log(y)
       log_z = np.log(z)
       
       log_t1 = p[1]*log_x + p[2]*log_y + p[3]*log_z
       log_t2 = p[5]*log_x + p[6]*log_y + p[7]*log_z
       
       # Clip the log values to prevent exp overflow
       # exp(700) overflows float64.
       max_log = 700.0
       min_log = -700.0
       
       log_t1 = np.clip(log_t1, min_log, max_log)
       log_t2 = np.clip(log_t2, min_log, max_log)
       
       t1 = p[0] * np.exp(log_t1)
       t2 = p[4] * np.exp(log_t2)
       
       t3 = p[8] * (x + y + z)
       t4 = p[9]
       
       result = t1 + t2 + t3 + t4
       
       if not np.all(np.isfinite(result)):
           result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
           
       return result
