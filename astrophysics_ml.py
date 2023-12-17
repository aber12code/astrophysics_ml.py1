

import numpy as np
import matplotlib.pyplot as plt 
# Generate synthetic astronomical data
np.random.seed(42)  # Set seed for reproducibility
num_points = 50
x_observation = np.sort(10 * np.random.rand(num_points))
y_observation = 2 * x_observation + 1 + np.random.randn(num_points)

# Plot the synthetic data
plt.scatter(x_observation, y_observation, label='Observations')
plt.xlabel('X (Astronomical Feature)')
plt.ylabel('Y (Measured Quantity)')
plt.title('Synthetic Astronomical Observation')
plt.legend()
plt.show()

# Use linear regression to fit a line to the data
A = np.vstack([x_observation, np.ones_like(x_observation)]).T
m, c = np.linalg.lstsq(A, y_observation, rcond=None)[0]

# Plot the fitted line
plt.scatter(x_observation, y_observation, label='Observations')
plt.plot(x_observation, m * x_observation + c, 'r', label='Fitted Line')
plt.xlabel('X (Astronomical Feature)')
plt.ylabel('Y (Measured Quantity)')
plt.title('Linear Regression in Astrophysics')
plt.legend()
plt.show()

# Print the slope and intercept of the fitted line
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
