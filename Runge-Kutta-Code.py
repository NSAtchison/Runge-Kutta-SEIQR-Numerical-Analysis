import numpy as np
import matplotlib.pyplot as plt

# SEIQR model derivatives
def seiqr_derivatives(S, E, I, Q, R, alpha, sigma, gamma, delta, N):
    s_prime = -alpha * S * I / N
    e_prime = alpha * S * I / N - sigma * E
    i_prime = sigma * E - gamma * I
    q_prime = gamma * I - delta * Q
    r_prime = delta * Q
    return np.array([s_prime, e_prime, i_prime, q_prime, r_prime])

def runge_kutta(S0, E0, I0, Q0, R0, steps, dt, alpha, sigma, gamma, delta):
    # Runge-Kutta 4th order (RK4) method
    S, E, I, Q, R = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], E[0], I[0], Q[0], R[0] = S0, E0, I0, Q0, R0

    N = S0 + E0 + I0 + Q0 + R0  # Total population

    for t in range(1, steps):
        y = np.array([S[t-1], E[t-1], I[t-1], Q[t-1], R[t-1]])
        
        k1 = dt * seiqr_derivatives(*y, alpha, sigma, gamma, delta, N)
        k2 = dt * seiqr_derivatives(*(y + k1 / 2), alpha, sigma, gamma, delta, N)
        k3 = dt * seiqr_derivatives(*(y + k2 / 2), alpha, sigma, gamma, delta, N)
        k4 = dt * seiqr_derivatives(*(y + k3), alpha, sigma, gamma, delta, N)
        
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        S[t], E[t], I[t], Q[t], R[t] = y_next

    return S, E, I, Q, R

def graph_results(S, E, I, Q, R, T, steps):
    # Plotting results
    time = np.linspace(0, T, steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time, S, label='Susceptible')
    plt.plot(time, E, label='Exposed')
    plt.plot(time, I, label='Infected')
    plt.plot(time, Q, label='Quarantined')
    plt.plot(time, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIQR Model Simulation (RK4 Method)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Parameters
    alpha = float(input("Enter the rate of transmission: ")) # Transmission rate
    sigma = float(input("Enter the rate of exposed to infected: ")) # Rate of exposed to infectious
    gamma = float(input("Enter the rate of infected to quarantined: ")) # Rate of infected to quarantined
    delta = float(input("Enter the rate of quarantined to recovered: ")) # Rate of quarantined to recovered

    S0 = int(input("Enter the initial susceptible population: "))
    E0 = int(input("Enter the initial exposed population: "))
    I0 = int(input("Enter the initial infected population: "))
    Q0 = int(input("Enter the initial quarantined population: "))
    R0 = int(input("Enter the initial recovered population: "))

    # Time parameters
    T = int(input("Enter the amount of days you want to simulate "))      # Total time (days)
    dt = float(input("Enter the time step to be used by Runge-Kutta Method: "))      # Time step
    steps = int(T / dt)  # Number of time steps

    S, E, I, Q, R = runge_kutta(S0, E0, I0, Q0, R0, steps, dt, alpha, sigma, gamma, delta)
    graph_results(S, E, I, Q, R, T, steps)


# Using the special variable 
# __name__
if __name__=="__main__":
    main()


