import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Define the Lorenz system equations
def lorenz(t, X, sigma=10, beta=8/3, rho=28):
    x, y, z = X
    dxdt = -sigma * x + sigma * y
    dydt = rho * x - y - x * z
    dzdt = -beta * z + x * y
    return [dxdt, dydt, dzdt]

# Define the response system equations
def lorenz_response(t, X, X1, sigma=10, beta=8/3, rho=28):
    x, y, z = X
    dxdt = -sigma * x + sigma * y
    dydt = rho * X1 - y - X1 * z
    dzdt = -beta * z + X1 * y
    return [dxdt, dydt, dzdt]

# Add AWGN noise to the signal
def add_awgn_noise(signal, SNR_dB):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / 10**(SNR_dB / 10)
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    return signal + noise

# Time parameters
dt = 0.001
maxtime = 20
t_eval = np.arange(0, maxtime, dt)

# Initial conditions for the drive system
xt0 = [17 + np.random.rand(), 18 + np.random.rand(), 20 + np.random.rand()]

# Solve the drive system
sol_drive = solve_ivp(lorenz, [0, maxtime], xt0, t_eval=t_eval, method='RK45')
XD = sol_drive.y

# Add noise to the drive system
SNRdB = 100
XDv = add_awgn_noise(XD[0], SNRdB)

# Interpolate XDv for arbitrary time points
XDv_interp = interp1d(t_eval, XDv, kind='linear', fill_value="extrapolate")

# Initial conditions for the response system
xt0_response = [5 + np.random.rand(), 4 + np.random.rand(), 6 + np.random.rand()]

# Solve the response system
sol_response = solve_ivp(lambda t, X: lorenz_response(t, X, XDv_interp(t)), [0, maxtime], xt0_response, t_eval=t_eval, method='RK45')
XR = sol_response.y

# Plotting the results

# Error plots
labels = ['x', 'y', 'z']
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(t_eval, XD[i] - XR[i], 'k')
    plt.ylabel(labels[i])
plt.xlabel('time')
plt.show()

# Time Domain plots
plt.figure(figsize=(12, 10))
plt.subplot(423)
plt.plot(t_eval, XD[0])
plt.ylabel('xR')
plt.title('Drive System')
plt.subplot(425)
plt.plot(t_eval, XD[1])
plt.ylabel('yR')
plt.subplot(427)
plt.plot(t_eval, XD[2])
plt.ylabel('zR')
plt.xlabel('time')
plt.subplot(424)
plt.plot(t_eval, XR[0])
plt.ylabel('xD')
plt.title('Response System')
plt.subplot(426)
plt.plot(t_eval, XR[1])
plt.ylabel('yD')
plt.subplot(428)
plt.plot(t_eval, XR[2])
plt.ylabel('zD')
plt.xlabel('time')
plt.subplot(421)
plt.plot(t_eval, XD[0])
plt.ylabel('x+n')
plt.title('Drive Signal')
plt.show()

# D vs R plots
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(XD[0], XR[0])
plt.title('xR VS xD')
plt.subplot(132)
plt.plot(XD[1], XR[1])
plt.title('yR VS yD')
plt.subplot(133)
plt.plot(XD[2], XR[2])
plt.title('zR VS zD')
plt.show()

# 3D plot of the hyperplane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(XD[0], XD[1], XR[1])
ax.set_xlabel('x_D')
ax.set_ylabel('y_D')
ax.set_zlabel('y_R')
plt.show()
