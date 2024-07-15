import matplotlib.pyplot as plt


def line_plot_parameters(flap_mode, train_x, train_y, N_INITIAL_SAMPLES, save_directory=None):
    data = train_x.cpu().numpy()
    num_plots = train_x.shape[-1]

    fig, axs = plt.subplots(2, 1,
                            figsize=(5, 8),
                            constrained_layout=True)

    # Define titles for each dimension
    colours = ['blue', 'red', 'orange', 'green']
    if flap_mode == 'test':
        labels = ['Top', 'Bottom', 'Right', 'Left']
    elif flap_mode == 'LR_symmetric':
        labels = ['Top', 'Bottom', 'Right/Left']
    elif flap_mode == 'LR_only_symmetric':
        labels = ['Right/Left']

    # Loop through each dimension and create a 1D scatter plot
    for i in range(num_plots):
        # Scatter plot with all y values at 0 to create a 1D effect
        axs[0].plot(data[:, i], marker='.', color=colours[i], label=labels[i])

    if N_INITIAL_SAMPLES > 0:
        axs[0].axvspan(xmin=0, xmax=N_INITIAL_SAMPLES-1.0, color='red', alpha=0.3)
    axs[0].grid(True)  # Enable grid along x-axis for better visibility
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Angle values')
    axs[0].legend()

    # Plot iterations
    axs[1].plot(train_y, color='black', marker='.')
    axs[1].axvspan(xmin=0, xmax=N_INITIAL_SAMPLES - 1.0, color='red', alpha=0.3)
    axs[1].set_xlabel('Training iteration')
    axs[1].set_ylabel('Reward/Performance')

    # Show the plot
    if not save_directory:
        plt.show()
    else:
        plt.savefig(save_directory + 'lineplot.png', format='png', dpi=300)

