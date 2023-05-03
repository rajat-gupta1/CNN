import pdb
import time
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt    

DATA_TYPE = np.float32
EPSILON = 1e-12


def calculate_fan_in_and_fan_out(shape):
    """
    The function takes the shape of an entity (weights or filters) and returns 
    fan_in which is the number of neurons in the input layer and 
    fan_out which is the number of neurons in the output layers
    """

    if len(shape)<2:
        raise ValueError("Unable to calculate fan_in and fan_out with dimension less than 2")
    
    elif len(shape)==2:  # Weight of a Fully Connected Layer
        fan_in, fan_out = shape[0], shape[1]

    elif len(shape)==4:  # filter of a convolutional layer
        
        # Product of all layers except the last one (which is the output dimension)
        fan_in = np.prod(shape[:-1])

        # Product of all layers except the third one (which is the input dimension)
        fan_out = shape[-1] * np.prod(shape[:-2])

    else:
        raise ValueError(f"Shape {shape} not supported in calculate_fan_in_and_fan_out")
    return fan_in, fan_out


def xavier(shape, seed=None):
    """
    The function takes the shape of an entity (weights or filters) and returns
    an np array of the input shape and randomized values basis xavier initialisation
    """
    n_in, n_out = calculate_fan_in_and_fan_out(shape)

    # In case the seed is given
    if seed is not None:
        np.random.seed(seed)

    # initializing uniformly at random from [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    range = np.sqrt(6 / (n_in + n_out))
    weights = np.random.uniform(low=-range, high=range, size=shape)
    return weights

class InputValue:
    """
    The class is for the leaf nodes in the computational graph where no gradients are required
    """
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()

class Param:
    """
    The class is for weights and biases - The trainable parameters whose values need to be updated
    """  
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
'''

            
class Add: 
    """
    The class takes two matrices during initialisation and computes their element wise 
    sum during forward propagation and gradients wrt element wise addition during 
    backward propagation
    """  

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.sum(self.grad.reshape([-1, len(self.b.value)]), axis=0)            


class Mul:
    """
    The class takes two matrices during initialisation and computes their element wise 
    product during forward propagation and gradients wrt element wise product during 
    backward propagation
    """  

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value    

class VDot:
    """
    The class takes a vector and a matrix during initialisation and computes their matrix
    product during forward propagation and gradients wrt matrix multiplication during 
    backward propagation
    """  
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = np.matmul(self.a.value, self.b.value)

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += np.matmul(self.grad, self.b.value.T)
        if self.b.grad is not None:
            # todo
            self.b.grad += np.matmul(self.a.value.reshape(self.a.value.shape[0], 1), self.grad.reshape(1, self.grad.shape[0]))


class Sigmoid:
    """
    The class takes a vector/matrix during initialisation and computes sigmoid for each element
    during forward propagation and the gradients wrt sigmoid during backward propagation
    """

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = 1 / (1 + np.exp(-self.a.value))

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad * self.value * (1 - self.value)

class RELU:
    """
    The class takes a vector/matrix during initialisation and computes Relu for each element
    during forward propagation and the gradients wrt Relu during backward propagation
    """

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = np.maximum(0, self.a.value)

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad * (self.value > 0)

class SoftMax:
    """
    The class takes a vector during initialisation and computes Softmax for each element
    during forward propagation and the gradients wrt Softmax during backward propagation
    """

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = np.exp(self.a.value) / np.sum(np.exp(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad * self.value - self.value * np.sum(self.grad * self.value)

class Log:
    """
    The class takes a vector/matrix during initialisation and computes log of each element
    during forward propagation and the gradients wrt log for each element during backward 
    propagation
    """

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = np.log(self.a.value)

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad / self.a.value


class Aref:
    """
    The class is used to get some specific entry in a matrix. a is a matrix with shape
    (batch size, N) and idx is a vector containing the entry index and a differentiable.
    During forward propagation, a[batch_size, idx] is calculated and during backward
    propagation, derivatives wrt input matrix a are calculated.
    """

    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


class Accuracy:
    """
    The function checks if the predicted label is correct or not. During forward propagation, 
    the lable with maximum probability is taken and compared with the ground truth label
    """
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass

class Conv:
    """
    The class computes the result after applying a filter of given dimensions in the 
    forward propagation and calculates the resulting gradients using that filter 
    during backward propagation
    """

    def __init__(self, input_tensor, kernel, stride=1, padding=0):
        """
        param input_tensor: input tensor of size (height, width, in_channels)
        param kernel: convolving kernel of size (kernel_size, kernel_size, in_channels, out_channels),
                        only square kernels of size (kernel_size, kernel_size) are supported
        param stride: stride of convolution. Default: 1
        param padding: zero-padding added to both sides of the input. Default: 0
        """

        self.kernel = kernel
        self.input_tensor = input_tensor
        self.padding = padding
        self.stride = stride
        self.grad = None if kernel.grad is None and input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
         calculates self.value of size (output_height, output_width, out_channels)
        """

        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        output_height = int((height + 2 * self.padding - kernel_size) / self.stride) + 1
        output_width = int((width + 2 * self.padding - kernel_size) / self.stride) + 1

        # Creating a dummy matrix
        self.value = np.zeros((output_height, output_width, output_channels))

        for i in range(output_height):
            for j in range(output_width):
                for c in range(output_channels):
                    self.value[i, j, c] = np.sum(padded_input[(i * self.stride): (i * self.stride) + kernel_size, j * self.stride: (j * self.stride) + kernel_size, :] * self.kernel.value[:, :, :, c])

    def backward(self):
        """
         calculates gradient of kernel.grad and input_tensor
        """

        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        kernel_grad = np.zeros(self.kernel.value.shape)
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        input_grad = np.zeros(padded_input.shape)

        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                for c in range(output_channels):
                    i0 = i * self.stride
                    j0 = j * self.stride

                    # Adding values to kernel_grad and input_grad 
                    kernel_grad[:, :, :, c] += padded_input[i0 : i0 + kernel_size, j0 : j0 + kernel_size, :] * self.grad[i, j, c]
                    input_grad[i0:(i0 + kernel_size), j0:(j0 + kernel_size), :] += self.kernel.value[:, :, :, c] * self.grad[i, j, c]

        # Updating self.kernel.grad and self.input_tensor.grad
        if self.kernel.grad is not None:
            self.kernel.grad = self.kernel.grad + kernel_grad
        if self.input_tensor.grad is not None:
            self.input_tensor.grad = self.input_tensor.grad + input_grad[self.padding:(self.padding + height),
                                                              self.padding:(self.padding + width), :]


class MaxPool:
    """
    The class computes the result after applying a MaxPool filter of given dimensions in the 
    forward propagation and calculates the resulting gradients using that filter 
    during backward propagation
    """

    def __init__(self, input_tensor, kernel_size=2, stride=None):
        """
        param input_tensor: input tensor of size (height, width, in_channels)
        param kernel_size: the size of the window to take a max over. Default: 2
        param stride: the stride of the window. Default value is kernel_size
        """

        self.input_tensor = input_tensor
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
        calculates self.value of size (int(height / self.stride), int(width / self.stride), in_channels)
        """

        height, width, in_channels = self.input_tensor.value.shape
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        self.value = np.zeros((output_height, output_width, in_channels))
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    self.value[i, j, c] = np.max(self.input_tensor.value[i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, c])


    def backward(self):
        """
        calculates the gradient for input_tensor
        """

        height, width, in_channels = self.input_tensor.value.shape
        input_grad = np.zeros(self.input_tensor.value.shape)
        
        for c in range(in_channels):
            for i in range(height):
                for j in range(width):

                    # If this value is the max value used in the subsequent layer
                    if self.value[i // self.stride, j // self.stride, c] == self.input_tensor.value[i, j, c]:
                        input_grad[i, j, c] = self.grad[i // self.stride, j // self.stride, c]
                    else:
                        input_grad[i, j, c] = 0
        self.input_tensor.grad = self.input_tensor.grad + input_grad


class Flatten:
    """
    The class flattens (converts into a vector) the input during forward propagation 
    calculates the resulting gradients during backward propagation
    """
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.input_tensor.value.flatten()

    def backward(self):
        if self.input_tensor.grad is not None:
            self.input_tensor.grad += self.grad.reshape(self.input_tensor.value.shape)


class CNN:
    """
    This is the main class, which is used for creating the CNN object and running it 
    to a particular dataset
    """

    def __init__(self, num_labels=10):
        self.num_labels = num_labels

        # dictionary of trainable parameters
        self.params = {}

        # list of computational graph
        self.components = []

        # Creating different placeholders
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    def nn_unary_op(self, op, a):
        """
        helper function for creating a unary operation object and add it to the computational graph
        """

        unary_op = op(a)
        print(f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    def nn_binary_op(self, op, a, b):
        """
        helper function for creating a binary operation object and add it to the computational graph
        """

        binary_op = op(a, b)
        print(f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def conv_op(self, input_tensor, kernel, stride=1, padding=0):
        """
        helper function for Conv objects and to add them to the computational graph
        """

        conv = Conv(input_tensor, kernel, stride=stride, padding=padding)
        print(f"Append <{conv.__class__.__name__}> to the computational graph")
        self.components.append(conv)
        return conv

    def maxpool_op(self, input_tensor, kernel_size=2, stride=None):
        """
        helper function for MaxPool objects and to add them to the computational graph
        """

        maxpool = MaxPool(input_tensor, kernel_size=kernel_size, stride=stride)
        print(f"Append <{maxpool.__class__.__name__}> to the computational graph")
        self.components.append(maxpool)
        return maxpool

    def set_params_by_dict(self, param_dict: dict):
        """
        to create a dict of parameters with parameter names as keys and numpy arrays as values
        """

        # resetting params to an empty dict before setting new values
        self.params = {}

        # adding Param objects to the dictionary of trainable paramters with names and values
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def get_param_dict(self):
        """
        to return the dict of parameters
        """

        # Extracting trainable parameter values from the dict of Params
        param_dict = {
            "conv1_kernel": self.params["conv1_kernel"].value,
            "conv1_bias": self.params["conv1_bias"].value,
            "conv2_kernel": self.params["conv2_kernel"].value,
            "conv2_bias": self.params["conv2_bias"].value,
            "fc1_weight": self.params["fc1_weight"].value,
            "fc1_bias": self.params["fc1_bias"].value,
            "fc2_weight": self.params["fc2_weight"].value,
            "fc2_bias": self.params["fc2_bias"].value,
            "fc3_weight": self.params["fc3_weight"].value,
            "fc3_bias": self.params["fc3_bias"].value
        }
        return param_dict

    def init_params_with_xavier(self):
        """
        Method to initialise param_dict such that each key is mapped to a numpy array of the corresponding size
        """

        param_dict = {
            "conv1_kernel": xavier((5, 5, 3, 6)),
            "conv1_bias": np.zeros((6,)),
            "conv2_kernel": xavier((5, 5, 6, 16)),
            "conv2_bias": np.zeros((16,)),
            "fc1_weight": xavier((400, 120)),
            "fc1_bias": np.zeros((120,)),
            "fc2_weight": xavier((120, 84)),
            "fc2_bias": np.zeros((84,)),
            "fc3_weight": xavier((84, self.num_labels)),
            "fc3_bias": np.zeros((self.num_labels,)),
        }
        self.set_params_by_dict(param_dict)

    def build_computational_graph(self):
        """
        Function to build a computational graph skeleton to be used for training parameters
        """

        # Resetting computational graph to empty list
        self.components = []

        input_tensor = self.sample_placeholder

        # Adding each instance to self.components
        conv = self.conv_op(input_tensor, self.params["conv1_kernel"])
        sum = self.nn_binary_op(Add, conv, self.params["conv1_bias"])
        relu = self.nn_unary_op(RELU, sum)
        maxpool = self.maxpool_op(relu)
        conv = self.conv_op(maxpool, self.params["conv2_kernel"])
        sum = self.nn_binary_op(Add, conv, self.params["conv2_bias"])
        relu = self.nn_unary_op(RELU, sum)
        maxpool = self.maxpool_op(relu)
        flatten = self.nn_unary_op(Flatten, maxpool)
        vdot = self.nn_binary_op(VDot, flatten, self.params["fc1_weight"])
        add = self.nn_binary_op(Add, vdot, self.params["fc1_bias"])
        relu = self.nn_unary_op(RELU, add)
        vdot = self.nn_binary_op(VDot, relu, self.params["fc2_weight"])
        add = self.nn_binary_op(Add, vdot, self.params["fc2_bias"])
        relu = self.nn_unary_op(RELU, add)
        vdot = self.nn_binary_op(VDot, relu, self.params["fc3_weight"])
        add = self.nn_binary_op(Add, vdot, self.params["fc3_bias"])
        pred = self.nn_unary_op(SoftMax, add)
        return pred

    def cross_entropy_loss(self):
        """
        Constructing cross entropy loss using self.pred_placeholder and self.label_placeholder
        """

        label_prob = self.nn_binary_op(Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss

    def eval(self, X, y):
        """
        Method to evaluate the performance of the model
        """

        if len(self.components) == 0:
            raise ValueError("Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
        Using stochastic gradient descent, training the model, calculating the accuracy and loss
        and generating the graphs for the same. 
        """

        # creating sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params.values():
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                
                # Setting values to sample and label placeholders
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])

                self.forward()
                self.backward(self.loss_placeholder)
                self.sgd_update_parameter(alpha)

            # evaluating on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

        # Generating graphs
        plt.plot(range(t), train_acc, 'b', marker='o', label="Train Accuracy")  
        plt.plot(range(t), train_loss, 'r', marker='o', label="Train Loss")     
        plt.xlabel('epochs') 
        plt.ylabel('Accuracy/Loss') 
        plt.title('Training Accuracy and Loss vs epochs')  
        plt.legend(loc="upper right")
        plt.show()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss): 
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        for p in self.params.values():
            p.value = p.value - lr * p.grad



def main():
    # Importing the dataset
    data = np.load('./cifar10_data/sub_data.npz')

    # Normalizing the dataset
    X = np.float32(data['imgs'])/255.

    # Reshape the valid image data to (idx, h, w, channel)
    X = X.reshape(10000, 32, 32, 3)
    y = np.float32(data['labels'])

    # for simplicity, focussing on the first four classes there are 4000 images in total
    sub_idx = np.where(y<=3)[0]
    X = X[sub_idx]
    y = y[sub_idx]

    # split in to train an test set
    train_x, test_x = X[:3000], X[3000:]
    train_y, test_y = y[:3000], y[3000:]
    model = CNN(num_labels=4)
    model.init_params_with_xavier()
    model.fit(train_x, train_y, 0.01, 10)

if __name__ == "__main__":
    main()
