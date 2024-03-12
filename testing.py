import numpy as np
import matplotlib.pyplot as plt
import os

saving_dir = "C:\\Users\\mzhou9\\OneDrive - University of California, San Francisco (1)\\PhD projects\\qualifiying exam\\figures"

# Generate spike times (example data)
num_trials = 50
num_iterations = 10
freq = 60 # hz
trial_duration = 6 # seconds
number_frames = freq * trial_duration
F = []
for num in range(num_iterations):
    raw_mul = []
    for n in range(num_trials):
        raw = np.random.normal(0, 0.8, size=(number_frames,))
        raw_mul.append(raw)
    F.append(np.mean(raw_mul, axis=0))

F[0][60] = 1
F[0][100] = 0.8
F[1][30] = 0.5
F[1][100] = 1.3
F[2][50] = 0.65
F[2][140] = 0.65
F[3][70] = 0.3
F[3][90] = 0.9
F[3][130] = 0.7
F[4][40] = 1
F[4][80] = 0.8
F[4][130] = 0.6

F[5][60] = 1
F[5][100] = 0.8
F[5][190] = 0.9
F[5][230] = 0.6
F[5][280] = 0.8
F[6][50] = 0.65
F[6][140] = 0.65
F[6][190] = 0.9
F[6][230] = 0.6
F[6][280] = 0.8
F[7][40] = 1
F[7][80] = 0.8
F[7][130] = 0.6
F[7][210] = 0.5
F[7][260] = 0.7

# Plot PSTH
fig, axs = plt.subplots(num_iterations, 1, figsize=(2, num_iterations/2))
x = np.linspace(0, trial_duration, number_frames)
for a in range(num_iterations):
    axs[a].plot(x, F[a])
    axs[a].set_ylim([0, 1.1])

fig.savefig(os.path.join(saving_dir, "Simulated_traces_dca1.eps"), format='eps')

