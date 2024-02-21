import numpy as np
import time
import plotly.graph_objects as go
from activation_functions import ReLU, dReLU, softmax

class SFW_NN:
    def __init__(self, X_train, y_train, X_test, y_test, initfactor, batch_size=12, epochs=500):
        '''Initializes the network parameters'''
        self.start = time.time()
        self.input_data = X_train
        self.target_data = y_train
        self.test_data = X_test
        self.target_test =  y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.initfactor = initfactor

        self.loss_history = []
        self.accuracy_history = []

        #self.test_loss_history = []
        self.test_accuracy_history = []

        self.epoch_decimal = []
        self.cpu_time = []

        self.W1_values = []
        self.W2_values = []

        # Initialize weights and biases
        self.init_weights_and_biases()

    def plot_init_weights(self):

        ##### W1 #####

        # Define the boundaries of the l∞-ball
        lower_boundary = -self.Linf_radius1
        upper_boundary = self.Linf_radius1

        # Plot the weights and the boundaries of the l∞-ball
        plt.figure(figsize=(10, 6))

        colors = ['bo', 'go', 'ro', 'mo']
        for i in range(self.W1.T.shape[1]):
            plt.plot(self.W1.T[:, i], colors[i], label=f'Weights of Neuron {i+1}')


        # Create arrays with the same length as self.W1.T for the boundaries
        lower_boundary_array = np.full_like(self.W1.T, lower_boundary)
        upper_boundary_array = np.full_like(self.W1.T, upper_boundary)

        plt.plot(lower_boundary_array[0], 'k--', label='Lower boundary')
        plt.plot(upper_boundary_array[0], 'k--', label='Upper boundary')
        plt.title('W1 weights and l∞-Ball constraint')
        plt.xlabel('Weights index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        # Set the Y-axis limits to show the upper and lower boundaries
        ylim_min = min(lower_boundary.min() * 2, self.W1.T.min())
        ylim_max = max(upper_boundary.max() * 2, self.W1.T.max())
        plt.ylim(ylim_min, ylim_max)

        plt.show()

        ##### W2 ####
        # Define the boundaries of the l∞-ball
        lower_boundary = -self.Linf_radius2
        upper_boundary = self.Linf_radius2

        # Plot the weights and the boundaries of the l∞-ball
        plt.figure(figsize=(10, 6))

        colors = ['bo', 'go', 'ro']  # list of colors for the three neurons
        for i in range(self.W2.shape[1]):
            plt.plot(self.W2[:, i], colors[i], label=f'Weights of Neuron {i+1}')


        # Create arrays with the same length as self.W2 for the boundaries
        lower_boundary_array = np.full_like(self.W2, lower_boundary)
        upper_boundary_array = np.full_like(self.W2, upper_boundary)

        plt.plot(lower_boundary_array[0], 'k--', label='Lower boundary')
        plt.plot(upper_boundary_array[0], 'k--', label='Upper boundary')
        plt.title('W2 weights and l∞-Ball constraint')
        plt.xlabel('Weights index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        # Set the Y-axis limits to show the upper and lower boundaries
        ylim_min = min(lower_boundary.min() * 2, self.W2.min())
        ylim_max = max(upper_boundary.max() * 2, self.W2.max())
        plt.ylim(ylim_min, ylim_max)

        plt.show()

    def init_weights_and_biases(self):
        '''Initializes the network weights and biases randomly'''

        '''The Glorot (or Xavier) uniform initialization is a method to initialize the weights of the neural network.
        The weights are drawn from a distribution with zero mean and a specific variance.
        For the uniform distribution, the weights are drawn from a range [-limit, limit], where limit is
        sqrt(6 / (fan_in + fan_out)), with fan_in representing the number of input units in the weight tensor and fan_out is the number of output units.'''

        # Get the dimensions of the input, hidden, and output layers
        n_input = self.input_data.shape[1]
        n_hidden = 64
        n_output = self.target_data.shape[1]

        # Calculate the limit for the uniform distribution
        limit1 = np.sqrt(6. / (n_input + n_hidden))
        limit2 = np.sqrt(6. / (n_hidden + n_output))

        '''
        Calculating the expected L2 norm of this uniform distribution over the interval [-limit, limit] is equivalent to
        calculating the root mean square (RMS) of the uniform distribution, which can be obtained with the formula:

        RMS = sqrt((a^2 + b^2 + ab) / 3)

        where a and b represent the interval of the uniform distribution. So for Glorot uniform initialization, a is -limit and b is limit.
        Replacing a and b with their values, the expected L2 norm becomes:

        RMS = sqrt(((-limit)^2 + (limit)^2 + (-limit)*(limit)) / 3)
        = sqrt((2 * limit^2) / 3)

        So, the expected L2-norm of Glorot uniform initialized values would be sqrt((2 * limit^2) / 3), where limit is sqrt(6 / (fan_in + fan_out)).
        '''

        # Compute the expected L2 norm
        self.expected_l2_norm1 = np.sqrt((2 * limit1**2) / 3)
        self.expected_l2_norm2 = np.sqrt((2 * limit2**2) / 3)

        # Initialize the weights with values drawn from a uniform distribution with range [-limit1, limit1]
        self.W1 = np.random.uniform(-limit1, limit1, (n_input, n_hidden))
        self.W2 = np.random.uniform(-limit2, limit2, (n_hidden, n_output))

        '''
        Each layer is constrained into an L-infinity ball with L2-diameter equal to 2 * 'self.initfactor' times the expected
        L2-norm of the Glorot uniform initialized values
        '''
        # Compute the radius
        self.Linf_radius1 = self.initfactor * self.expected_l2_norm1
        self.Linf_radius2 = self.initfactor * self.expected_l2_norm2

        # Clip the weights to this radius
        self.W1 = np.clip(self.W1, -self.Linf_radius1, self.Linf_radius1)
        self.W2 = np.clip(self.W2, -self.Linf_radius2, self.Linf_radius2)

        # Save the initial weights
        self.W1_values.append(self.W1[:, 0].copy())  # storing the weights of the first neuron of the hidden layer (R4 vector)
        self.W2_values.append(self.W2[0, :].copy().T)  # storing the weights from the first neuron of the hidden layer to output layer (R3 vector)

        print(" W1 radius", self.Linf_radius1)
        print(" W2 radius", self.Linf_radius2)


        '''
        This approach can be understood as an application of projected gradient descent, where the weights are first
        initialized and then projected onto the l∞-ball of the desired size.

        The l∞-ball constraint can help control the spread of values in the weights, which can be useful in preventing
        excessive weights values which might lead to problems like exploding gradients during training.'''

        # Initialize the bias as 0 or as really small value
        self.b1 = np.zeros(n_hidden)
        self.b2 = np.zeros(n_output)

        print("W1:",self.W1.shape)
        print("W2:",self.W2.shape)

        #self.plot_init_weights()

    def shuffle_data(self):
        '''Shuffles the input and target data in unison'''
        indices = np.arange(self.input_data.shape[0])
        np.random.shuffle(indices)
        self.input_data = self.input_data[indices]
        self.target_data = self.target_data[indices]

    def feedforward(self, batch_input, batch_target):
        '''Performs the forward pass through the network'''
        self.layer1_pre_activation = np.dot(batch_input, self.W1) + self.b1
        self.layer1_post_activation = ReLU(self.layer1_pre_activation)

        self.layer2_pre_activation = np.dot(self.layer1_post_activation, self.W2) + self.b2
        self.output = softmax(self.layer2_pre_activation)

        self.error = self.output - batch_target

    def compute_direction(self, g, radius):

        '''Computes the direction s using the Frank-Wolfe algorithm with Linf norm ball constraints.
        When the constrain is a polytope (e.g. a L-infinity ball), we know, by means of the fundamental theorem
        of linear programming that one of the vertices is solution of the linear program.
        It allows us to search for the optimal solution within the vertices, rather than exploring the entire feasible region
        '''
        s = np.zeros_like(g)
        for i in range(len(g)):
            if g[i] > 0:
                s[i] = -radius  # minimize in this dimension
            else:
                s[i] = radius  # maximize in this dimension
        return s

    def backpropagation(self, batch_input, epoch):
        '''Performs the backward pass through the network'''
        error_derivative = (1/self.batch_size) * self.error

        # Compute gradients for layer 2 (output layer) weights and biases
        dW2 = np.dot(self.layer1_post_activation.T, error_derivative)
        db2 = np.sum(error_derivative, axis=0)

        # Compute gradient for layer 1 weights and biases
        dcost_dz1 = np.dot(error_derivative, self.W2.T)
        dz1_da1 = dReLU(self.layer1_pre_activation)
        dcost_da1 = dcost_dz1 * dz1_da1
        dW1 = np.dot(batch_input.T, dcost_da1)
        db1 = np.sum(dcost_da1, axis=0)

        # Update weights and biases using the Frank-Wolfe algorithm
        self.s_W2 = self.compute_direction(dW2.flatten(), self.Linf_radius2)
        self.s_b2 = self.compute_direction(db2.flatten(), self.Linf_radius2)
        self.s_W1 = self.compute_direction(dW1.flatten(), self.Linf_radius1)
        self.s_b1 = self.compute_direction(db1.flatten(), self.Linf_radius1)

        alpha = 2 / (2 + epoch)  # Step size parameter

        self.W2 = self.W2 + alpha * (self.s_W2.reshape(self.W2.shape) - self.W2)
        self.b2 = self.b2 + alpha * (self.s_b2.reshape(self.b2.shape) - self.b2)
        self.W1 = self.W1 + alpha * (self.s_W1.reshape(self.W1.shape) - self.W1)
        self.b1 = self.b1 + alpha * (self.s_b1.reshape(self.b1.shape) - self.b1)


        self.W1_values.append(self.W1[:, 0].copy())  # storing the weights of the first neuron of the hidden layer (R4 vector)
        self.W2_values.append(self.W2[0, :].copy().T)  # storing the weights from the first neuron of the hidden layer to output layer (R3 vector)


    def train(self):
        '''Trains the network using mini-batch gradient '''
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for batch_index in range(self.input_data.shape[0] // self.batch_size - 1):
                # Shuffle the data at the beginning of each epoch
                self.shuffle_data()

                # Extract the current mini-batch from the input and target data
                start_index = batch_index * self.batch_size
                end_index = (batch_index + 1) * self.batch_size
                batch_input = self.input_data[start_index:end_index]
                batch_target = self.target_data[start_index:end_index]

                # Perform a forward pass through the network
                self.feedforward(batch_input, batch_target)

                # Perform a backward pass through the network (backpropagation)
                self.backpropagation(batch_input, epoch)

                # Update the total loss and accuracy for this epoch
                epoch_loss += np.mean(self.error ** 2)
                epoch_accuracy += np.count_nonzero(np.argmax(self.output, axis=1) == np.argmax(batch_target, axis=1)) / self.batch_size

                # Calculate training loss
                self.loss_history.append(np.mean(self.error ** 2))


                # Calculate accuracy for the test dataset
                self.feedforward(self.test_data, self.target_test)
                #self.test_loss_history.append(np.mean(self.error ** 2))

                correct_predictions = np.count_nonzero(np.argmax(self.output, axis=1) == np.argmax(self.target_test, axis=1))
                total_predictions = self.test_data.shape[0]

                self.test_accuracy_history.append(correct_predictions * 100 / total_predictions)

                # Save epoch number and CPU time
                decimal = batch_index / (self.input_data.shape[0] // self.batch_size - 1)
                self.epoch_decimal.append(epoch + decimal)
                self.cpu_time.append(time.time() - start_time)


            # Add the average loss and accuracy for this epoch to the history
            self.accuracy_history.append(epoch_accuracy * 100 / (self.input_data.shape[0] // self.batch_size))

            # Print the average loss and accuracy for this epoch
            print(f'Time {time.time() - self.start:.5f}s, Epoch {epoch + 1}:             Loss = {epoch_loss / (self.input_data.shape[0] // self.batch_size)},            Accuracy = {epoch_accuracy * 100 / (self.input_data.shape[0] // self.batch_size)}%')


    def plot_learning_curves(self):
        '''Plots the learning curves for loss and accuracy over epochs'''
        #### Plot training accuracy curve for testing
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot accuracy over epochs
        ax.plot(self.accuracy_history)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy curve (to be removed)")

        plt.tight_layout()
        plt.show()


        #### Plot comprehensive curves
        from scipy.signal import savgol_filter

        # Create subplots
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Define the color for the curves
        curve_color = 'blue'

        # Plot loss over epochs
        ax[0, 0].plot(self.epoch_decimal, self.loss_history, label='Loss', alpha=0.2, color=curve_color)
        smoothed_loss = savgol_filter(self.loss_history, window_length=11, polyorder=3)
        ax[0, 0].plot(self.epoch_decimal, smoothed_loss, label='Smoothed Loss', color=curve_color)
        ax[0, 0].set_ylabel("Training")

        # Plot loss over CPU time
        ax[0, 1].plot(self.cpu_time, self.loss_history, label='Loss', alpha=0.2, color=curve_color)
        ax[0, 1].plot(self.cpu_time, smoothed_loss, label='Smoothed Loss', color=curve_color)
        ax[0, 1].set_xlabel("CPU Time")

        # Plot accuracy over epochs
        ax[1, 0].plot(self.epoch_decimal, self.test_accuracy_history, label='Accuracy', alpha=0.2, color=curve_color)
        smoothed_accuracy = savgol_filter(self.test_accuracy_history, window_length=11, polyorder=3)
        ax[1, 0].plot(self.epoch_decimal, smoothed_accuracy, label='Smoothed Accuracy', color=curve_color)
        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("Test")

        # Plot accuracy over CPU time
        ax[1, 1].plot(self.cpu_time, self.test_accuracy_history, label='Accuracy', alpha=0.2, color=curve_color)
        ax[1, 1].plot(self.cpu_time, smoothed_accuracy, label='Smoothed Accuracy', color=curve_color)
        ax[1, 1].set_xlabel("CPU Time")

        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        ax[0, 0].set_title("Loss over Epoch")
        ax[0, 1].set_title("Loss over CPU Time")
        ax[1, 0].set_title("Accuracy over Epoch")
        ax[1, 1].set_title("Accuracy over CPU Time")

        plt.suptitle("Comprehensive Training and Test learning curves", fontsize=16)

        plt.show()


    def plot_2D_update(self):
        # extract the first two weights of the first neuron of W1 for each update
        W1_first_two_weights = [weights[:2] for weights in self.W1_values]
        W1_x = [weights[0] for weights in W1_first_two_weights]
        W1_y = [weights[1] for weights in W1_first_two_weights]

        # do the same for W2
        W2_first_two_weights = [weights[:2] for weights in self.W2_values]
        W2_x = [weights[0] for weights in W2_first_two_weights]
        W2_y = [weights[1] for weights in W2_first_two_weights]

        # create L inf ball (a square in 2D)
        x_1 = np.array([self.Linf_radius1, self.Linf_radius1, -self.Linf_radius1, -self.Linf_radius1, self.Linf_radius1])
        y_1 = np.array([self.Linf_radius1, -self.Linf_radius1, -self.Linf_radius1, self.Linf_radius1, self.Linf_radius1])

        x_2 = np.array([self.Linf_radius2, self.Linf_radius2, -self.Linf_radius2, -self.Linf_radius2, self.Linf_radius2])
        y_2 = np.array([self.Linf_radius2, -self.Linf_radius2, -self.Linf_radius2, self.Linf_radius2, self.Linf_radius2])

        ball_trace_1 = go.Scatter(x=x_1, y=y_1, mode='lines', name='L inf ball')
        ball_trace_2 = go.Scatter(x=x_2, y=y_2, mode='lines', name='L inf ball')

        # create the plots
        trace1 = go.Scatter(x=W1_x, y=W1_y, mode='lines+markers+text', line=dict(dash='dot'), text=list(range(len(W1_x))), name='W1 weights')
        trace2 = go.Scatter(x=W2_x, y=W2_y, mode='lines+markers+text', line=dict(dash='dot'), text=list(range(len(W2_x))), name='W2 weights')


        layout1 = go.Layout(title='Showing weight update for weights 1 and 2 of neuron 1 of W1 (between input and hidden layer)', xaxis=dict(title='Weight 1'), yaxis=dict(title='Weight 2'))
        layout2 = go.Layout(title='Showing weight update for weights 1 and 2 of neuron 1 of W2 (between hidden and output layer)', xaxis=dict(title='Weight 1'), yaxis=dict(title='Weight 2'))

        # create a Figure for W1 and add traces to it
        fig1 = go.Figure(data=[trace1, ball_trace_1], layout=layout1)
        fig1.update_layout(showlegend=False)
        # plot the figure
        fig1.show()

        # create a Figure for W2 and add traces to it
        fig2 = go.Figure(data=[trace2, ball_trace_2], layout=layout2)
        fig2.update_layout(showlegend=False)
        # plot the figure
        fig2.show()



    def test(self):
        '''Tests the trained network using test data'''
        # Use the test input and target data to perform a forward pass through the network
        self.feedforward(self.test_data, self.target_test)
        #print(self.output)

        # Compute the accuracy by comparing the output of the network to the target data
        correct_predictions = np.count_nonzero(np.argmax(self.output, axis=1) == np.argmax(self.target_test, axis=1))
        total_predictions = self.test_data.shape[0]
        accuracy = correct_predictions / total_predictions

        print(f'Accuracy: {100 * accuracy}%')

