import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import constant, orthogonal
from tensorflow.keras.layers import Activation, Dense, Layer


class BaseModel(Model):
    """A base model framework, which implements save, restore, and pretraining methods.

    Methods
    =======
    - :meth:`save_state`: Save the current state.
    - :meth:`restore_state`: Restore a stored state.
    - :meth:`pretrain_init`: Initialize pretraining.
    - :meth:`pretrain_init_rec`: Recursively initialize the pretraining for prenorm layers in the model.
    - :meth:`pretrain_next`: Finds a prenorm layer that has received updates but has not yet stopped pretraining.
    - :meth:`pretrain_next_rec`: Recursively finds a prenorm layer that has received updates but has not yet stopped
        pretraining.
    - :meth:`pretrain`: Pretrains the model.
    """

    def save_state(self, path: str):
        """Save the current state.

        :param path: The desired save path.
        """

        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    def restore_state(self, path: str):
        """Restore a stored state.

        :param path: The path to a stored state.
        """

        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))

    def pretrain_init(self):
        """Initialize pretraining."""

        self.pretrain_init_rec(self, self.name)

    @staticmethod
    def pretrain_init_rec(model: Model, name: str):
        """Recursively initialize the pretraining for prenorm layers in the model.

        :param model: The model to initialize.
        :param name: The name of the model.
        """

        for layer in model.layers:
            if isinstance(layer, Model):
                # Recursively look for a prenorm layer.
                BaseModel.pretrain_init_rec(layer, f"{name}/{layer.name}")
            elif isinstance(layer, PreNormLayer):
                layer.start_updates()

    def pretrain_next(self):
        """Finds a prenorm layer that has received updates but has not yet stopped pretraining.

        Used in a pretraining loop, has no direct interpretation. See :func:`model_trainer.pretrain` for it's usage.

        :return: A prenorm layer that has received updates but has not yet stopped pretraining.
        """

        return self.pretrain_next_rec(self, self.name)

    @staticmethod
    def pretrain_next_rec(model: Model, name: str):
        """Recursively finds a prenorm layer that has received updates but has not yet stopped pretraining.

        :param model: The model to search for prenorm layers.
        :param name: The name of the model.
        :return: A prenorm layer that has received updates but has not yet stopped pretraining.
        """

        for layer in model.layers:
            if isinstance(layer, Model):
                # Recursively look for a prenorm layer.
                result = BaseModel.pretrain_next_rec(layer, f"{name}/{layer.name}")
                if result is not None:
                    return result
            elif isinstance(layer, PreNormLayer) and layer.waiting_updates and layer.received_updates:
                layer.stop_updates()
                return layer, f"{name}/{layer.name}"
        return None

    def pretrain(self, *args, **kwargs):
        """Pretrains the model.

        :param args: Arguments used to call the model.
        :param kwargs: Keyword arguments used to call the model.
        :return: True whenever a layer has received updates.
        """

        try:
            self.call(*args, **kwargs)
            # This successfully finishes when no layer receives updates.
            return False
        except PreNormException:
            # This occurs whenever a layer receives updates.
            return True


class GCNN(BaseModel):
    """The graph convolutional neural network (GCNN) model.

    Extends the BaseModel class.

    Methods
    =======
    - :meth:`build`: Builds the model.
    - :meth:`call`: Calls the model on inputs and returns outputs.
    - :meth:`pad_output`: Pads the model output in case the input shape differs between batch instances.

    :ivar emb_size: The embedding size of each feature vector.
    :ivar cons_feats: The number of features for each constraint or added cut (row).
    :ivar edge_feats: The number of features for each edge (both constraint and cut candidate).
    :ivar var_feats: The number of features for each variable (column).
    :ivar cut_feats: The number of features for each cut candidate.
    :ivar cons_embedding: Constraint embedding module.
    :ivar cons_edge_embedding: Constraint edge embedding module.
    :ivar var_embedding: Variable embedding module.
    :ivar cut_embedding: Cut candidate embedding module.
    :ivar cut_edge_embedding: Cut edge embedding module.
    :ivar conv_v_to_c: Variable to constraint convolution module.
    :ivar conv_c_to_v: Constraint to variable convolution module.
    :ivar conv_v_to_k: Variable to cut convolution module.
    :ivar output_module: The output module.
    :ivar variables_topological_order: A list of the variables for save/restore.
    :ivar input_signature: Input signature for compilation.
    """

    def __init__(self):
        super().__init__()

        self.emb_size = 64
        self.cons_feats = 4
        self.edge_feats = 1
        self.var_feats = 9
        self.cut_feats = 6

        # Constraint embedding.
        self.cons_embedding = Sequential([PreNormLayer(n_units=self.cons_feats),
                                          Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal),
                                          Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal)])

        # Constraint edge embedding.
        self.cons_edge_embedding = Sequential([PreNormLayer(self.edge_feats)])

        # Variable embedding.
        self.var_embedding = Sequential([PreNormLayer(n_units=self.var_feats),
                                         Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal),
                                         Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal)])

        # Cut candidate embedding.
        self.cut_embedding = Sequential([PreNormLayer(n_units=self.cut_feats),
                                         Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal),
                                         Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal)])

        # Cut edge embedding.
        self.cut_edge_embedding = Sequential([PreNormLayer(self.edge_feats)])

        # Graph convolutions.
        self.conv_v_to_c = PartialGraphConvolution(self.emb_size, relu, orthogonal, from_v=True)
        self.conv_c_to_v = PartialGraphConvolution(self.emb_size, relu, orthogonal)
        self.conv_v_to_k = PartialGraphConvolution(self.emb_size, relu, orthogonal, from_v=True)

        # Output.
        self.output_module = Sequential([Dense(units=self.emb_size, activation=relu, kernel_initializer=orthogonal),
                                         Dense(units=1, activation=None, kernel_initializer=orthogonal,
                                               use_bias=False)])

        # Build the model right away.
        self.build([(None, self.cons_feats), (2, None), (None, self.edge_feats), (None, self.var_feats),
                    (None, self.cut_feats)])

        # Used for save/restore.
        self.variables_topological_order = [v.name for v in self.variables]

        # Save input signature for compilation.
        self.input_signature = [(tf.TensorSpec(shape=[None, self.cons_feats], dtype=tf.float32),
                                 tf.TensorSpec(shape=[2, None], dtype=tf.int32),
                                 tf.TensorSpec(shape=[None, self.edge_feats], dtype=tf.float32),
                                 tf.TensorSpec(shape=[None, self.var_feats], dtype=tf.float32),
                                 tf.TensorSpec(shape=[None, self.cut_feats], dtype=tf.float32),
                                 tf.TensorSpec(shape=[None], dtype=tf.int32),
                                 tf.TensorSpec(shape=[None], dtype=tf.int32),
                                 tf.TensorSpec(shape=[None], dtype=tf.int32)), tf.TensorSpec(shape=[], dtype=tf.bool)]

    def build(self, input_shapes: list):
        """Builds the model.

        :param input_shapes: The input shapes to use for building the model, array of the from [c_shape, ei_shape,
        e_shape, v_shape, k_shape].
        """

        c_shape, ei_shape, e_shape, v_shape, k_shape = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.cons_embedding.build(c_shape)
            self.cons_edge_embedding.build(e_shape)
            self.var_embedding.build(v_shape)
            self.cut_embedding.build(k_shape)
            self.cut_edge_embedding.build(e_shape)
            self.conv_v_to_c.build((emb_shape, ei_shape, e_shape, emb_shape))
            self.conv_c_to_v.build((emb_shape, ei_shape, e_shape, emb_shape))
            self.conv_v_to_k.build((emb_shape, ei_shape, e_shape, emb_shape))
            self.output_module.build(emb_shape)
            self.built = True

    def call(self, inputs, training):
        """Calls the model using the specified input.

        Accepts stacked mini-batches, in which case the number of candidate cuts per sample has to be provided,
        and the output consists of a padded dense tensor.

        Input is of the form [*cons_feats, cons_edge_inds,
        cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons, n_vars, n_cuts*],
        with the following parameters:

        - *cons_feats*: 2D constraint feature tensor of size (*n_cons, n_cons_features*).
        - *cons_edge_inds*: 2D edge index tensor of size (*n_cons_edges, n_cons_features*).
        - *cons_edge_feats*: 2D edge feature tensor of size (*n_cons_edges, n_edge_features*).
        - *var_feats*: 2D variable feature tensor of size (*n_vars, n_var_features*).
        - *cut_feats*: 2D cut candidate feature tensor of size (*n_cuts, n_cut_features*).
        - *cut_edge_inds*: 2D edge index tensor of size (*n_cut_edges, n_cut_features*).
        - *cut_edge_feats*: 2D edge feature tensor of size (*n_cut_edges, n_edge_features*).
        - *n_cons*: 1D tensor that contains the number of constraints for each sample.
        - *n_vars*: 1D tensor that contains the number of variables for each sample.
        - *n_cuts*: 1D tensor that contains the number of cut candidates for each sample.

        :param inputs: A list of input tensors.
        :param training: True if in training mode.
        :return: The model output, a vector in case of a single sample, a padded dense tensor in case of a stacked
            mini-batch.
        """

        cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons, \
        n_vars, n_cuts = inputs
        n_cons_total = tf.reduce_sum(n_cons)
        n_vars_total = tf.reduce_sum(n_vars)
        n_cuts_total = tf.reduce_sum(n_cons)

        # Embeddings.
        cons_feats = self.cons_embedding(cons_feats)
        cons_edge_feats = self.edge_embedding(cons_edge_feats)
        var_feats = self.var_embedding(var_feats)
        cut_feats = self.cut_embedding(cut_feats)
        cut_edge_feats = self.cut_edge_embedding(cut_edge_feats)

        # Partial graph convolutions.
        cons_feats = self.conv_v_to_c((cons_feats, cons_edge_inds, cons_edge_feats, var_feats, n_cons_total), training)
        cons_feats = relu(cons_feats)

        var_feats = self.conv_c_to_v((cons_feats, cons_edge_inds, cons_edge_feats, var_feats, n_vars_total), training)
        var_feats = relu(var_feats)

        cut_feats = self.conv_v_to_k((cut_feats, cut_edge_inds, cut_edge_feats, var_feats, n_cuts_total), training)
        cut_feats = relu(cut_feats)

        # Output.
        output = self.output_module(cut_feats)
        output = tf.reshape(output, [1, -1])

        if n_cuts.shape[0] > 1:
            output = self.pad_output(output, n_cuts)

        return output

    @staticmethod
    def pad_output(output, n_vars, pad_value=-1e8):
        """Splits the output by sample and pads with very low logits.

        :param output:
        :param n_vars:
        :param pad_value:
        :return:
        """
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split(value=output, num_or_size_splits=n_vars_per_sample, axis=1, )
        output = tf.concat(
            [tf.pad(x, paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]], mode='CONSTANT', constant_values=pad_value)
             for x in output], axis=0)

        return output


class PreNormLayer(Layer):
    """A pre-normalization layer, used to normalize an input layer to zero mean and unit variance in order to speed-up
    and stabilize GCNN training.

    This layer extends Keras' layer object, from which all other layers descend as well. The layer's parameters are
    computed during the pretraining phase.

    Methods
    =======
    - :meth:`build`: Called when the layer is initialized, before the layer is called.
    - :meth:`call`: The layer's call function.
    - :meth:`start_updates`: Initialize the pretraining phase.
    - :meth:`update_params`: Update parameters in pretraining.
    - :meth:`stop_updates`: End pretraining and fix the layers's parameters.

    :ivar shift: The shifting weights.
    :ivar scale: Tha scaling weights.
    :ivar n_units: The number of input units.
    :ivar waiting_updates: True if in pretraining.
    :ivar received_updates: True if in pretraining and the layer has received parameter updates.
    :ivar mean: The current mean value, used for pretraining.
    :ivar var: The current variance value, used for pretraining.
    :ivar m2: The current sum of squared differences from the mean, used for pretraining.
    :ivar count: The current number of samples, used for pretraining.
    """

    def __init__(self, n_units, shift=True, scale=True):
        super().__init__()

        if shift:
            # Initialize a shifting weight for each input unit.
            self.shift = self.add_weight(name=f'{self.name}/shift', shape=(n_units,), trainable=False,
                                         initializer=constant(value=np.zeros((n_units,)), dtype=tf.float32), )
        else:
            self.shift = None

        if scale:
            # Initialize a scaling weight for each input unit.
            self.scale = self.add_weight(name=f'{self.name}/scale', shape=(n_units,), trainable=False,
                                         initializer=constant(value=np.ones((n_units,)), dtype=tf.float32), )
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

        self.mean = None
        self.var = None
        self.m2 = None
        self.count = None

    def build(self, input_shapes):
        """Called when the layer is initialized, before the layer is called.

        :param input_shapes: An instance of TensorShape or a list of these objects.
        """

        self.built = True

    def call(self, input, *args, **kwargs):
        """The layer's call function.

        :param input: An input tensor.
        :return: The shifted and/or scaled input.
        """

        if self.waiting_updates:
            self.update_params(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input += self.shift

        if self.scale is not None:
            input *= self.scale
        return input

    def start_updates(self):
        """Initialize the pretraining phase."""

        self.mean = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_params(self, input):
        """Update parameters in pretraining.

        Uses an online mean and variance estimation algorithm suggested by [1]_,
        see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.


        References
        ==========
        .. [1] Chan, T. F., Golub, G. H., & LeVeque, R. J. (1982). Updating formulae and a pairwise algorithm for
            computing sample variances. COMPSTAT, 30–41. https://doi.org/10.1007/978-3-642-51461-6_3

        :param input: An input tensor.
        """

        assert self.n_units == 1 or input.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.shape[-1]}."

        # Compute sample mean and variance.
        input = tf.reshape(input, [-1, self.n_units])
        sample_mean = tf.reduce_mean(input, 0)
        sample_var = tf.reduce_mean((input - sample_mean) ** 2, axis=0)
        sample_count = tf.cast(tf.size(input=input) / self.n_units, tf.float32)

        # Update the sum of squared differences from the current mean (m2).
        delta = sample_mean - self.mean
        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        # Extract new mean and variance.
        self.count += sample_count
        self.mean += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """End pretraining and fix the layers's parameters."""

        if self.shift is not None:
            self.shift.assign(-self.mean)

        if self.scale is not None:
            self.var = tf.where(tf.equal(self.var, 0), tf.ones_like(self.var), self.var)
            self.scale.assign(1 / np.sqrt(self.var))

        del self.mean, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class PreNormException(Exception):
    """Used for pretraining, raised whenever a layer receives updates."""

    pass


class PartialGraphConvolution(Model):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, activation, initializer, from_v=False):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.from_v = from_v

        # feature layers
        self.feature_module_left = Sequential(
            [Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer)])
        self.feature_module_edge = Sequential(
            [Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)])
        self.feature_module_right = Sequential(
            [Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)])
        self.feature_module_final = Sequential([PreNormLayer(1, shift=False),  # normalize after summation trick
                                                Activation(self.activation), Dense(units=self.emb_size, activation=None,
                                                                                   kernel_initializer=self.initializer)])

        self.post_conv_module = Sequential([PreNormLayer(1, shift=False),  # normalize after convolution
                                            ])

        # output_layers
        self.output_module = Sequential(
            [Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
             Activation(self.activation),
             Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer), ])

    def build(self, input_shapes):
        l_shape, ei_shape, ev_shape, r_shape = input_shapes

        self.feature_module_left.build(l_shape)
        self.feature_module_edge.build(ev_shape)
        self.feature_module_right.build(r_shape)
        self.feature_module_final.build([None, self.emb_size])
        self.post_conv_module.build([None, self.emb_size])
        self.output_module.build([None, self.emb_size + (l_shape[1] if self.from_v else r_shape[1])])
        self.built = True

    def call(self, inputs, training):
        """
        Perfoms a partial graph convolution on the given bipartite graph.
        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)
        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.from_v:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        # compute joint features
        joint_features = self.feature_module_final(tf.gather(self.feature_module_left(left_features), axis=0,
                                                             indices=edge_indices[0]) + self.feature_module_edge(
            edge_features) + tf.gather(self.feature_module_right(right_features), axis=0, indices=edge_indices[1]))

        # perform convolution
        conv_output = tf.scatter_nd(updates=joint_features, indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
                                    shape=[scatter_out_size, self.emb_size])
        conv_output = self.post_conv_module(conv_output)

        # apply final module
        output = self.output_module(tf.concat([conv_output, prev_features, ], axis=1))

        return output
