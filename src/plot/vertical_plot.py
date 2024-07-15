import matplotlib.pyplot as plt
import numpy as np


def vertical_plot(train_x, train_y, N_INITIAL_SAMPLES, save_directory=None):
    data = train_x.cpu().numpy()
    num_plots = train_x.shape[-1]

    # Create the figure and axes for the subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 8))

    # Loop through each dimension and create a 1D scatter plot
    for i in range(num_plots):
        # Scatter plot with all y values at 0 to create a 1D effect
        axes[i].scatter(data[:, i], np.zeros(data.shape[0]), marker='x', color='blue')
        if N_INITIAL_SAMPLES > 0:
            axes[i].scatter(data[:N_INITIAL_SAMPLES, i], np.zeros(N_INITIAL_SAMPLES), marker='x', color='green')
        axes[i].set_ylim(-0.1, 0.1)  # Ensure the line looks horizontal
        axes[i].set_yticks([])  # Hide y-axis ticks
        axes[i].grid(True, axis='x')  # Enable grid along x-axis for better visibility

    # Set the xlabel on the bottom subplot
    axes[-1].set_xlabel('Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    if not save_directory:
        plt.show()
    else:
        plt.savefig(save_directory + 'vertical_plot.png', format='png', dpi=300)
