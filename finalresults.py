"""
# Final results: we only compare SFW with AdaSVRF
"""

def plot_learning_curves(nns, MAX_EPOCHS = 100, MAX_CPU_TIME = 2):
    col = 0
    colors = ["c", "m", "y", "orange"]
    from scipy.signal import savgol_filter

    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for nn in nns:
        '''Plots the learning curves for loss and accuracy over epochs'''

        #### Plot comprehensive curves

        # Define the color for the curves
        curve_color = colors[col]

        # CUT data to MAX_EPOCHS and MAX_CPU_TIME
        indices = np.where(np.asarray(nn.epoch_decimal) > MAX_EPOCHS)
        # Get the first index
        if len(indices[0]) > 0:
            CUT_EPOCHS = indices[0][0]
        else:
            CUT_EPOCHS = len(nn.epoch_decimal)

        indices = np.where(np.asarray(nn.cpu_time) > MAX_CPU_TIME)
        # Get the first index
        if len(indices[0]) > 0:
            CUT_CPU = indices[0][0]
        else:
            CUT_CPU = len(nn.epoch_decimal)

        # Plot loss over epochs
        ax[0, 0].plot(nn.epoch_decimal[:CUT_EPOCHS], nn.loss_history[:CUT_EPOCHS], label='Loss', alpha=0.2, color=curve_color)
        smoothed_loss = savgol_filter(nn.loss_history, window_length=50, polyorder=3)
        ax[0, 0].plot(nn.epoch_decimal[:CUT_EPOCHS], smoothed_loss[:CUT_EPOCHS], label='Smoothed Loss', color=curve_color)
        ax[0, 0].set_ylabel("Training")

        print(np.shape(nn.epoch_decimal))
        print(CUT_EPOCHS)
        print(np.shape(nn.loss_history))
        # Plot loss over CPU time
        ax[0, 1].plot(nn.cpu_time[:CUT_CPU], nn.loss_history[:CUT_CPU], label='Loss', alpha=0.2, color=curve_color)
        ax[0, 1].plot(nn.cpu_time[:CUT_CPU], smoothed_loss[:CUT_CPU], label='Smoothed Loss', color=curve_color)
        ax[0, 1].set_xlabel("CPU Time")

        # Plot accuracy over epochs
        ax[1, 0].plot(nn.epoch_decimal[:CUT_EPOCHS], nn.test_accuracy_history[:CUT_EPOCHS], label='Accuracy', alpha=0.2, color=curve_color)
        smoothed_accuracy = savgol_filter(nn.test_accuracy_history, window_length=50, polyorder=3)
        ax[1, 0].plot(nn.epoch_decimal[:CUT_EPOCHS], smoothed_accuracy[:CUT_EPOCHS], label='Smoothed Accuracy', color=curve_color)
        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("Test")

        # Plot accuracy over CPU time
        ax[1, 1].plot(nn.cpu_time[:CUT_CPU], nn.test_accuracy_history[:CUT_CPU], label='Accuracy', alpha=0.2, color=curve_color)
        ax[1, 1].plot(nn.cpu_time[:CUT_CPU], smoothed_accuracy[:CUT_CPU], label='Smoothed Accuracy', color=curve_color)
        ax[1, 1].set_xlabel("CPU Time")

        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        col = col + 1


    ax[0, 0].set_title("Loss over Epoch")
    ax[0, 1].set_title("Loss over CPU Time")
    ax[1, 0].set_title("Accuracy over Epoch")
    ax[1, 1].set_title("Accuracy over CPU Time")

    # Create custom legend lines
    from matplotlib.lines import Line2D  #
    legend_lines = [Line2D([0], [0], color=colors[i], lw=2, label=type(nns[i]).__name__.split("_")[0]) for i in range(col)]

    # Set a single legend on the side using the custom lines
    fig.legend(handles=legend_lines, loc='center', prop={'size': 16})
    plt.show()

plot_learning_curves([SFW_iris, AdaSVRF_iris], MAX_EPOCHS = 20, MAX_CPU_TIME = 2)

plot_learning_curves([SFW_obesity, AdaSVRF_obesity], MAX_EPOCHS = 20, MAX_CPU_TIME = 2)
