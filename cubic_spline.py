from scipy.interpolate import CubicSpline
import torch


class TorchCubicSpline:
    def __init__(self, x_data, y_data, bc_type="not-a-knot", extrapolate=True):
        # x_data and y_data are 1D numpy arrays
        # Compute the spline coefficients using SciPy
        CS_scipy = CubicSpline(x_data, y_data, bc_type=bc_type, extrapolate=extrapolate)

        # Store knots and coefficients
        self.x_data = torch.tensor(CS_scipy.x)
        self.c_coeffs = torch.tensor(CS_scipy.c)  # Shape (4, n_intervals)
        self.extrapolate = extrapolate
        self.bc_type = bc_type

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        # Ensure x is within the valid range
        x_min = self.x_data[0]
        x_max = self.x_data[-1]

        if not self.extrapolate:
            x = torch.clamp(x, min=x_min, max=x_max)  # Clamp x to boundary values

        # Find the interval indices for each x
        indices = torch.bucketize(x, self.x_data, right=False) - 1
        indices = indices.clamp(min=0, max=self.c_coeffs.shape[1] - 1)

        # Get coefficients for each interval
        c3 = self.c_coeffs[3, indices]  # Constant term
        c2 = self.c_coeffs[2, indices]  # Linear term
        c1 = self.c_coeffs[1, indices]  # Quadratic term
        c0 = self.c_coeffs[0, indices]  # Cubic term

        # Compute the offset dx
        dx = x - self.x_data[indices]

        # Evaluate the cubic polynomial
        y = c3 + c2 * dx + c1 * dx**2 + c0 * dx**3

        return y

    def derivative(self):
        # Compute the derivative coefficients
        deriv_coeffs = torch.empty_like(self.c_coeffs)
        deriv_coeffs[0] = 0  # there is no cubic term etc
        deriv_coeffs[1] = 3 * self.c_coeffs[0]
        deriv_coeffs[2] = 2 * self.c_coeffs[1]
        deriv_coeffs[3] = self.c_coeffs[2]

        # Create a new instance for the derivative spline
        deriv_spline = TorchCubicSpline.__new__(
            TorchCubicSpline
        )  # no need to call __init__ since coefficients are known already
        deriv_spline.x_data = self.x_data
        deriv_spline.c_coeffs = deriv_coeffs
        deriv_spline.extrapolate = self.extrapolate
        deriv_spline.bc_type = self.bc_type
        return deriv_spline
