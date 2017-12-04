import numpy as np

from progressbar import ProgressBar

from activation import Activation

# https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
# for reference.

# TODO - maybe add a debug flag but it would probably always be on anyways.

# TODO - probably need to modify the code when we decide on the structure of inputs and outputs. 
# Right now assuming that X.shape == (T, num_features)

def Convert1DTo2D(vector):
    # Ensure that the vector is indeed 1D
    assert len(vector.shape) == 1
    return vector.reshape(vector.shape[0], 1)

def Convert2DTo1D(vector):
    # Ensure that the vector is indeed 2D
    assert len(vector.shape) == 2
    assert vector.shape[1] == 1
    return vector.flatten()



class RNN:
    def __init__(self,
                 input_layer_size,
                 state_layer_size, state_layer_activation,
                 output_layer_size, output_layer_activation,
                 epochs = 100,
                 bptt_truncate=None,
                 learning_rule='bptt',
                 kernel=None,
                 eta=0.001,
                 rand=None,
                 verbose=0):
        """
        Notes:
            U - weight matrix from input into hidden layer.
            W - weight matrix from hidden layer to hidden layer.
            V - weight matrix from hidden layer to output layer.

        Inputs:
            input_size:
                Size of the input vector. We expect a 2D numpy array, so this should be X.shape[1]

            state_layer_size:
                State layer size.

            state_layer_activation:
                A string. Refer to activation.py

            output_size:
                Size of the output vector. We expect a 2D numpy array, so this should be Y.shape[1]

            output_layer_activation:
                A string. Refer to activation.py

            epochs(opt):
                Number of epochs for a single training sample.

            learning_rule(opt):
                Choose between 'bptt' and 'modified'

            bptt_truncate(opt):
                If left at None, back propagation through time will be applied for all time steps. 

                Otherwise, a value for bptt_truncate means that 
                bptt will only be applied for at most bptt_truncate steps.

                Only considered when learning_rule == 'bptt'

            kernel(opt):
                # TODO - fill this
                Only considered when learning_rule == 'modified'

            eta (opt):
                Learning rate. Initialized to 0.001.

            rand (opt):
                Random seed. Initialized to None (no random seed).

            verbose (opt):
                Verbosity: levels 0 - 2

        Outputs:
            None
        """
        np.random.seed(rand)

        self.learning_rule = learning_rule.lower()

        if self.learning_rule == 'bptt':
            self.gradient_function = self.bptt
        elif self.learning_rule == 'modified':
            self.gradietn_function = self.modified_learning_rule
        else:
            raise ValueError

        self.input_layer_size = input_layer_size

        self.state_layer_size = state_layer_size
        self.state_layer_activation = state_layer_activation
        self.state_activation = Activation(state_layer_activation)

        self.output_layer_size = output_layer_size
        self.output_layer_activation = output_layer_activation
        self.output_activation = Activation(output_layer_activation)

        self.epochs = epochs

        self.kernel = kernel
        self.bptt_truncate = bptt_truncate

        # U - weight matrix from input into state layer.
        # W - weight matrix from state layer to state layer.
        # V - weight matrix from state layer to output layer.
        self.U = np.random.uniform(-np.sqrt(1./input_layer_size),
                                    np.sqrt(1./input_layer_size), 
                                    (state_layer_size, input_layer_size))
        self.V = np.random.uniform(-np.sqrt(1./state_layer_size),
                                    np.sqrt(1./state_layer_size),
                                    (output_layer_size, state_layer_size))
        self.W = np.random.uniform(-np.sqrt(1./state_layer_size),
                                    np.sqrt(1./state_layer_size),
                                    (state_layer_size, state_layer_size))
        self.state_bias = np.zeros((state_layer_size,1))
        self.output_bias = np.zeros((output_layer_size, 1))


        self.eta = eta
        self.verbose = verbose
        self.show_progress_bar = verbose > 0


    def fit(self, X_train, y_train):
        """
        Notes:

        Inputs:
            X_train:
                Training inputs. Expect a list with numpy arrays of size (input_layer_size, N) where N is the number of samples.

            Y_train:
                Training outputs. Expect a list with numpy arrays of size (output_layer_size, N) where N is the number of samples.

        Outputs:
            None
        """
        eta = self.eta

        if self.show_progress_bar:
            bar = ProgressBar(max_value = self.epochs)

        for epoch in range(self.epochs):
            for x, y in zip(X_train, y_train):
                dLdU, dLdV, dLdW, dLdOb, dLdSb = self.gradient_function(x, y)
                self.U -= eta * dLdU
                self.V -= eta * dLdV
                self.W -= eta * dLdW
                self.output_bias -= dLdOb
                self.state_bias -= dLdSb

            if self.show_progress_bar:
                bar.update(epoch)

    
    def forward_propagation(self, x):
        """
        Inputs:
            x:
                Expect size (T, input_layer_size), where T is the length of time.
        Outputs:
            o:
                The activation of the output layer.
            s:
                The activation of the hidden state. 
        """
        T = x.shape[0]

        s = np.zeros((T + 1, self.state_layer_size))
        o = np.zeros((T, self.output_layer_size))
        s_linear = np.zeros((T + 1, self.state_layer_size))
        o_linear = np.zeros((T, self.output_layer_size))

        state_bias = Convert2DTo1D(self.state_bias)
        output_bias = Convert2DTo1D(self.output_bias)
        
        for t in np.arange(T):
            state_linear = np.dot(self.U, x[t]) + np.dot(self.W, s[t-1]) + state_bias
            s_linear[t] = state_linear
            s[t] = self.state_activation.activate(state_linear)
            output_linear = np.dot(self.V, s[t]) + output_bias
            o[t] = self.output_activation.activate(output_linear)
            o_linear[t] = output_linear
        return (o, s, s_linear, o_linear)

    def modified_learning_rule(self, x, y):
        """ 
            Output:
                dLdU:
                    Gradient for U matrix
                dLdV:
                    Gradient for V matrix
                dLdW:
                    Gradient for W matrix
                dLdOb:
                    Gradient for output layer bias
                dLdSb:
                    Gradient for state layer bias
            TODO - implement this
        """
        raise NotImplementedError

    def bptt(self, x, y):
        """
            Output:
                dLdU:
                    Gradient for U matrix
                dLdV:
                    Gradient for V matrix
                dLdW:
                    Gradient for W matrix
                dLdOb:
                    Gradient for output layer bias
                dLdSb:
                    Gradient for state layer bias
        """
        # TODO - numpy likes to provide 1D matrices instead of 2D, and unfortunately
        # we need 2D matrices. Therefore we have a lot of converting 1D to 2D matrices
        # and we might want to clean that later somehow...

        # TODO - also this can probably be cleaned more.

        T = len(y)
        assert T == len(x)
        
        if self.bptt_truncate is None:
            bptt_truncate = T
        else:
            bptt_truncate = self.bptt_truncate

        o, s, s_linear, o_linear = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        dLdOb = np.zeros(self.output_bias.shape)
        dLdSb = np.zeros(self.state_bias.shape)

        num_dU_additions = 0
        num_dVdW_additions = 0

        delta_o = o - y
        for t in reversed(range(T)):
            # Backprop the error at the output layer
            g = delta_o[t]
            o_linear_val = o_linear[t]
            state_activation = s[t]

            g = Convert1DTo2D(g)
            o_linear_val = Convert1DTo2D(o_linear_val)
            state_activation = Convert1DTo2D(state_activation)

            g = g * self.output_activation.dactivate(o_linear_val)
            dLdV += np.dot(g, state_activation.T)
            dLdOb += g
            num_dU_additions += 1
            g = np.dot(self.V.T, g)

            # Backpropagation through time for at most bptt truncate steps
            for bptt_step in reversed(range(max(0, t - bptt_truncate),  t + 1)):
                state_linear = s_linear[bptt_step]
                state_activation_prev = s[bptt_step - 1]
                x_present = x[t]
                
                state_linear = Convert1DTo2D(state_linear)
                state_activation_prev = Convert1DTo2D(state_activation_prev)
                x_present = Convert1DTo2D(x_present)

                g = g  * self.state_activation.dactivate(state_linear)
                dLdW += np.dot(g, state_activation_prev.T)
                dLdU += np.dot(g, x_present.T)
                dLdSb += g
                num_dVdW_additions += 1

                g = g * np.dot(self.W.T, g)
        return [dLdU/num_dU_additions, 
                dLdV/num_dVdW_additions, 
                dLdW/num_dVdW_additions, 
                dLdOb/num_dU_additions, 
                dLdSb/num_dVdW_additions]

    def predict(self, X):
        """
        Inputs:
            X:
                Training inputs. Expect a list with numpy arrays of size (input_layer_size, N) where N is the number of samples.

        Outputs:
            predictions
        """
        predictions = []
        for x in X:
           o, _, _, _ = self.forward_propagation(x)
           predictions.append(o)
        return predictions

    def score(self, X, Y):
        """
        Inputs:
            X:
                Training inputs. Expect a list with numpy arrays of size (input_layer_size, N) where N is the number of samples.

            Y:
                Training outputs. Expect a list with numpy arrays of size (output_layer_size, N) where N is the number of samples.

        Outputs:
            MSE  
        """
        predictions = self.predict(X)
        mses = []
        for prediction, y in zip(predictions, Y):
           mses.append(np.mean((predictions - y)**2))
        return np.mean(mses)
                   

if __name__ == "__main__":
    # Example usage
    T = 10
    #x = np.array([[1,1], [2,2], [3,3], [4,4]])
    x = np.random.normal(0, 1, (T, 2))
    y = x[1:]
    x = x[:-1]

    input_size = x.shape[1]
    state_layer_size = 8
    output_size = y.shape[1]
    state_layer_activation = 'sigmoid'
    output_layer_activation = 'linear'

    rnn = RNN(input_size, state_layer_size, state_layer_activation, 
              output_size, output_layer_activation, 
              bptt_truncate=None, eta = 0.001, epochs=50000, verbose=1)

    X_train = [x]
    y_train = [y]
    o, s, _, _ = rnn.forward_propagation(x)
    print "Before training:"
    print "Outputs:"
    print o
    print "MSE:"
    print rnn.score(X_train, y_train)

    rnn.fit(X_train, y_train)

    o, s, _, _ = rnn.forward_propagation(x)
    print "\nAfter training:"
    print "Outputs:"
    print o
    print "MSE:"
    print rnn.score(X_train, y_train)

