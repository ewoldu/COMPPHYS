import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import scipy

class Ising_Models:
    def __init__(self,length,width):
        self.dim = (length,width)
        self.num_configs = 2**(length*width)
        self.datapoints = 2**17

    def get_num_configs(self):
        return self.num_configs

    def get_batch_size(self):
        if self.datapoints > 8192:
            return self.datapoints//8192
        else:
            return self.datapoints

    def make_bins(self):
        return tf.range(0,self.num_configs,2,dtype=tf.float32)

    def make_data(self):
        if self.num_configs < 2**17:
            self.datapoints = self.num_configs
            return tf.convert_to_tensor(np.arange(self.num_configs)[np.newaxis].T)
        return tf.convert_to_tensor(np.random.choice(2**(length*width), size=2**17)[np.newaxis].T)
    
    def neighbors_sum(self,lattice):
        '''
        Sums the spins of the lattice points at four neighbor sites to site (i,j).
            Takes into account the size of the lattice in 
            terms of number of rows (i) and columns (j),
            thus implementing periodic boundary conditions.
        '''
        return tf.roll(lattice,1,0) + tf.roll(lattice,-1,0) + tf.roll(lattice,1,1) + tf.roll(lattice,-1,1)

    def lattice_energy(self, lattice, J=1):
        mask = np.zeros(self.dim)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if (i + j) % 2 == 0:
                    mask[i][j] = 1
        mask = tf.constant(mask,dtype=tf.int32)
        #print(mask)
        #print(lattice)
        neighbors = self.neighbors_sum(lattice)
        #print(neighbors)
        return tf.math.scalar_mul(-J,tf.math.reduce_sum(lattice * neighbors * mask))

    def energy(self,nums):
        flat_dims = tf.reduce_prod(self.dim)
        new_nums = tf.bitcast(nums,tf.int32) + flat_dims**2
        new_nums = tf.repeat(tf.expand_dims(new_nums,axis=1),repeats=flat_dims,axis=1)
        #print(new_nums)
        shifts = tf.repeat(tf.expand_dims(tf.range(flat_dims),axis=0),repeats=new_nums.shape[0],axis=0)
        #print(shifts)
        bin_nums = tf.bitwise.right_shift(new_nums, shifts)
        #print(bin_nums)
        lattices = tf.reshape(bin_nums, [-1,*self.dim]) % 2
        #print(lattices)
        lattices = tf.where(lattices == 1,tf.ones_like(lattices),-tf.ones_like(lattices))
        #print(lattices)
        return tf.bitcast(tf.map_fn(self.lattice_energy, lattices),tf.float32)

# Creating a custom layer with keras API.
def Coupling(input_shape):
    output_dim = 256
    reg = 0.01
    input = keras.layers.Input(shape=(input_shape,))

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super().__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(2) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.

        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    @tf.function
    def kl_loss(self,x):
        x_new = tf.cast(x,dtype=tf.float32)
        #print(x_new)
        total = x_new.shape[0]
        edges = system.make_bins()
        #print(edges)
        y = tfp.stats.histogram(x_new,edges)
        #print(y)
        midpoints = (edges[:-1] + edges[1:]) // 2  # Midpoints of the bins
        #print(midpoints)
        energy_probs = tf.exp(-system.energy(midpoints) / (scipy.constants.k * 300))
        #print(f"Energy probabilities: {energy_probs}")

        # Calculate the KL divergence
        probs = y / total  # P(x) as probabilities
        kl_div = tf.reduce_sum(probs * tf.math.log(probs / energy_probs))
        print(f"KL Divergence: {kl_div}")
        
        return kl_div

    def train_step(self, data):
        with tf.GradientTape() as tape:

            #loss = self.log_loss(data)
            loss = self.kl_loss(data)

        # for var in self.trainable_variables:
        #     print(var.name, var.trainable)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        #loss = self.log_loss(data)
        loss = self.kl_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

if __name__ == "__main__":
    system = Ising_Models(4,4)
    data = system.make_data()
    model = RealNVP(num_coupling_layers=6)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    history = model.fit(data, batch_size=system.get_batch_size(), epochs=30, verbose=2, validation_split=0.2)

    

