import numpy as np
import matplotlib.pyplot as plt


# Parameters

N = 100  # Number of particles
m = 1.0  # Mass
gamma = 1  # Friction coefficient
k = 4.0  # Spring constant
kb = 1.0  # Boltzmann Constant
T = 1.0  # Temperature
dt = 0.01  # Time step, with this choice is guarantee that the integral is good
total_time = 100  # Total simulation time


# Initial conditions
x0 = np.zeros(N)  # Initial positions


# Function to generate Gaussian white noise


def generate_noise(dt, size):
    return np.sqrt(2 * kb * T / dt / m / gamma) * np.random.normal(size=size)


# Initialize arrays, everything to zero

time = np.arange(0, total_time, dt)
x = np.zeros((N, len(time)))


# Set initial conditions
x[:, 0] = x0


# n times iterations
# quadratic potential
# for i in range(1, len(time)):
#    noise = generate_noise(dt, N)
#    x[:, i] = x[:, i - 1] - k/(m*gamma) * x[:, i - 1] * dt + noise*dt

# second potential
for i in range(1, len(time)):
    noise = generate_noise(dt, N)
    x[:, i] = (
        x[:, i - 1]
        - 2 * k * (x[:, i - 1] ** 2 - 1) * 2 * x[:, i - 1] / (m * gamma) * dt
        + noise * dt
    )


def boltz(x):
    return 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)


def potenziale(x):
    return k * (x**2 - 1) ** 2


x_values = np.linspace(-2.0, 2.0, 100)
# y_values = boltz(x_values)
y_values = potenziale(x_values)


# Plot the results
plt.figure(figsize=(10, 6))
for j in range(1):
    plt.plot(time, x[j, :], label=f"Particle {j + 1}")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()


final_positions = x[:, N]
num_bins = 30
plt.hist(
    final_positions,
    bins=num_bins,
    density=True,
    alpha=0.7,
    color="blue",
)
plt.plot(x_values, y_values, label="Function: $y=x^2$")
plt.xlabel("Final Position")
plt.ylabel("Frequency")
plt.title("Histogram of Final Positions")
plt.show()
