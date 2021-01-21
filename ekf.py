import numpy as np 

def ekf(m, x, filterstep, error_cov, count):

    R = np.diag([.0001, .0001])
    I = np.eye(4)
    
    # Project the state ahead
    x[0] = x[0] + filterstep*x[3]*np.cos(x[2])
    x[1] = x[1] + filterstep*x[3]*np.sin(x[2])
    x[2] = (x[2]+ np.pi) % (2.0*np.pi) - np.pi
    x[3] = x[3]

    # Calculate the Jacobian of the Dynamic Matrix A
    a13 = filterstep*x[3]*np.sin(x[2])
    a14 = filterstep*np.cos(x[2])
    a23 = filterstep*x[3]*np.cos(x[2])
    a24 = filterstep*np.sin(x[2])
    JA = np.array([[1.0, 0.0, a13, a14],
                    [0.0, 1.0, a23, a24],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

    #Process noise cov
    Q = np.diag([0.03, 0.03, 0.03, 0.03]) 
    
    # Project the error covariance ahead
    error_cov = np.matmul(np.matmul(JA, error_cov), JA.T) + Q

    # Measurement Function
    hx = np.array([x[0], x[1]])

    if count%1==0: #every nth step pretend there is a measurement
        JH = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]])
    else: # every other step
        JH = np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])  

    S = np.matmul(np.matmul(JH, error_cov), JH.T) + R
    K = np.matmul(np.matmul(error_cov, JH.T), np.linalg.inv(S.astype('float64')))

    Z = np.array(m)
    y = Z - hx 
    x = x + np.matmul(K, y)
    # Update the error covariance
    error_cov = np.matmul((I - np.matmul(K, JH)), error_cov)
    return x, error_cov