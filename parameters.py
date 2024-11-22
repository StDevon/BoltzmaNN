import numpy as np

# The mass of the mother particle (MP)
m1 = 1
# The mass of the daughter particle (DP)
m2 = 0.5

# m1 = 1.776 # (tau) GeV
# m2 = 0.10565 # (muon) GeV

# The mass of the DM particle
mDM = 1e-10  # GeV

# PQ symmetry breaking scale
fa = 2e8  # GeV
# corresponds to the freeze out

# Ratio of the two masses
massRatio = m2 / m1

# Number of massless particle degrees of freedom
g_x = 1.0
# Number of MP dofs
g_m1 = 2.0
# Number of DP dofs
g_m2 = 2.0

# Coupling of MP-DP-x
yMDx = 1

# Decay width of MP -> DP + x
Gamma = yMDx**2 * m1**3 * (1 - (m2 / m1) ** 2) ** 3 / 64 / np.pi / fa**2  # GeV
# Msquared for the decay

Msq = yMDx**2 * m1**4 * (1 - (m2 / m1) ** 2) ** 2 / 4 / fa**2  # GeV^2
