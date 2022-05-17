"""This module contains the framework for the graph convolutional neural network model (GCNN).

Summary
=======
This module provides the framework for the GCNN used for cut selection. The methods in this module are based the code
by [1]_.

Classes
========
- :class:`BaseModel`: A base model framework, which implements save, restore, and pretraining methods.
- :class:`GCNN`: The graph convolutional neural network (GCNN) model.
- :class:`PreNormLayer`: A pre-normalization layer, used to normalize an input layer to zero mean and unit variance.
- :class:`PreNormException`: Exception used for pretraining.
- :class:`PartialGraphConvolution`: A partial bipartite graph convolution.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import Constant
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

        with open(path, 'wb') as file:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), file)

    def restore_state(self, path: str):
        """Restore a stored state.

        :param path: The path to a stored state.
        """

        with open(path, 'rb') as file:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(file))

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
        self.var_feats = 14
        self.cut_feats = 6

        # Constraint embedding.
        self.cons_embedding = Sequential([PreNormLayer(n_units=self.cons_feats),
                                          Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                                name='cons_emb_1'),
                                          Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                                name='cons_emb_2')])

        # Constraint edge embedding.
        self.cons_edge_embedding = Sequential([PreNormLayer(self.edge_feats)])

        # Variable embedding.
        self.var_embedding = Sequential([PreNormLayer(n_units=self.var_feats),
                                         Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                               name='var_emb_1'),
                                         Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                               name='var_emb_2')])

        # Cut candidate embedding.
        self.cut_embedding = Sequential([PreNormLayer(n_units=self.cut_feats),
                                         Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                               name='cut_emb_1'),
                                         Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal',
                                               name='cut_emb_2')])

        # Cut edge embedding.
        self.cut_edge_embedding = Sequential([PreNormLayer(self.edge_feats)])

        # Graph convolutions.
        self.conv_v_to_c = PartialGraphConvolution(self.emb_size, name='cons_conv', from_v=True)
        self.conv_c_to_v = PartialGraphConvolution(self.emb_size, name='var_conv')
        self.conv_v_to_k = PartialGraphConvolution(self.emb_size, name='cut_conv', from_v=True)

        # Output.
        self.output_module = Sequential(
            [Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal', name='out_1'),
             Dense(units=1, activation='relu', kernel_initializer='orthogonal', name='out_2')])

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
                                 tf.TensorSpec(shape=[2, None], dtype=tf.int32),
                                 tf.TensorSpec(shape=[None, self.edge_feats], dtype=tf.float32),
                                 tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.int32)), tf.TensorSpec(shape=[], dtype=tf.bool)]

    def build(self, input_shapes: list):
        """Builds the model.

        Input is of the form [*c_shape*, *ei_shape*, *e_shape*, *v_shape*, *k_shape*], with the following parameters:

        - *c_shape*: The shape of the constraint feature matrix.
        - *ei_shape*: The shape of the edge index matrix.
        - *e_shape*: The shape of the edge feature matrix.
        - *v_shape*: The shape of the variable feature matrix.
        - *k_shape*: The shape of the cut feature matrix.

        :param input_shapes: The input shapes to use for building the model.
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

    def call(self, inputs, training: bool):
        """Calls the model using the specified input.

        Accepts stacked mini-batches, in which case the number of candidate cuts per sample has to be provided,
        and the output consists of a padded tensor of size (*n_samples*, *max_cuts*).

        Input is of the form [*cons_feats*, *cons_edge_inds*, *cons_edge_feats*, *var_feats*, *cut_feats*,
        *cut_edge_inds*, *cut_edge_feats*, *n_cons*, *n_vars*, *n_cuts*], with the following parameters:

        - *cons_feats*: 2D constraint feature tensor of shape (sum(*n_cons*), *n_cons_features*).
        - *cons_edge_inds*: 2D edge index tensor of shape (2, *n_cons_edges*).
        - *cons_edge_feats*: 2D edge feature tensor of shape (*n_cons_edges*, *n_edge_features*).
        - *var_feats*: 2D variable feature tensor of shape (sum(*n_vars*), *n_var_features*).
        - *cut_feats*: 2D cut candidate feature tensor of shape (sum(*n_cuts*), *n_cut_features*).
        - *cut_edge_inds*: 2D edge index tensor of shape (2, *n_cut_edges*).
        - *cut_edge_feats*: 2D edge feature tensor of shape (*n_cut_edges*, *n_edge_features*).
        - *n_cons*: The total number of constraints.
        - *n_vars*: The total number of variables.
        - *n_cuts*: The total number of cuts.

        :param inputs: The model input.
        :param training: True if in training mode.
        :return: The model output, a vector in case of a single sample, a padded tensor of shape (*n_samples,
            max_cuts*) in case of a stacked mini-batch.
        """

        (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
         n_vars, n_cuts) = inputs

        # Embeddings.
        cons_feats = self.cons_embedding(cons_feats)
        cons_edge_feats = self.cons_edge_embedding(cons_edge_feats)
        var_feats = self.var_embedding(var_feats)
        cut_feats = self.cut_embedding(cut_feats)
        cut_edge_feats = self.cut_edge_embedding(cut_edge_feats)

        # Partial graph convolutions.
        cons_feats = self.conv_v_to_c((cons_feats, cons_edge_inds, cons_edge_feats, var_feats, n_cons), training)
        var_feats = self.conv_c_to_v((cons_feats, cons_edge_inds, cons_edge_feats, var_feats, n_vars), training)
        cut_feats = self.conv_v_to_k((cut_feats, cut_edge_inds, cut_edge_feats, var_feats, n_cuts), training)

        # Output.
        output = self.output_module(cut_feats)
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
    - :meth:`stop_updates`: End pretraining and fix the layer's parameters.

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

    def __init__(self, n_units: int, shift=True, scale=True):
        super().__init__()

        if shift:
            # Initialize a shifting weight for each input unit.
            constant_init = Constant(value=0.0)
            self.shift = self.add_weight(name=f'{self.name}/shift', shape=(n_units,), trainable=False,
                                         initializer=constant_init)
        else:
            self.shift = None

        if scale:
            # Initialize a scaling weight for each input unit.
            constant_init = Constant(value=1.0)
            self.scale = self.add_weight(name=f'{self.name}/scale', shape=(n_units,), trainable=False,
                                         initializer=constant_init)
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

    def call(self, inputs, *args, **kwargs):
        """The layer's call function.

        :param inputs: An input tensor.
        :return: The shifted and/or scaled input.
        """

        if self.waiting_updates:
            self.update_params(inputs)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            inputs += self.shift

        if self.scale is not None:
            inputs *= self.scale
        return inputs

    def start_updates(self):
        """Initialize the pretraining phase."""

        self.mean = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_params(self, inputs):
        """Update parameters in pretraining.

        Uses an online mean and variance estimation algorithm suggested by [1]_,
        see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.


        References
        ==========
        .. [1] Chan, T. F., Golub, G. H., & LeVeque, R. J. (1982). Updating formulae and a pairwise algorithm for
            computing sample variances. COMPSTAT, 30–41. https://doi.org/10.1007/978-3-642-51461-6_3

        :param inputs: An input tensor.
        """

        assert self.n_units == 1 or inputs.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {inputs.shape[-1]}."

        # Compute sample mean and variance.
        inputs = tf.reshape(inputs, [-1, self.n_units])
        sample_mean = tf.reduce_mean(inputs, 0)
        sample_var = tf.reduce_mean((inputs - sample_mean) ** 2, axis=0)
        sample_count = tf.cast(tf.size(input=inputs) / self.n_units, tf.float32)

        # Update the sum of squared differences from the current mean (m2).
        delta = sample_mean - self.mean
        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        # Extract new mean and variance.
        self.count += sample_count
        self.mean += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """End pretraining and fix the layer's parameters."""

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
    """A partial bipartite graph convolution.

    Extends TensorFlow's Model class. The left/right terminology adopted in this class is based on the bipartite
    graph representation of a combinatorial optimization problem, with rows (constraints and cuts) on the left-hand
    side, and columns (variables) on the right-hand side. In this graph representation, rows and columns have an edge
    between them if the corresponding variable appears in the row or cut.

    The partial graph convolution operates on a set of rows, a set of variables, and a set of edges between the two.
    For each edge, the algorithm first computes a joint feature representation by passing all feature vectors -
    sending node, edge, and receiving node - through a 64-node layer without activation (basically a weighted sum).
    This joint feature representation is then scaled, passed through a ReLU activation, and passed through another
    64-node layer without activation. Then, for the receiving side, this joint feature representation is added to the
    feature representation of the corresponding node. The final, feature representation is scaled, concatenated with
    the initial receiving feature matrix. Finally, this result is passed through two 64-node layers with ReLU
    activation.

    Methods
    =======
    - :meth:`build`: Builds the model.
    - :meth:`call`: Calls the model on inputs and returns outputs.

    :ivar emb_size: The embedding size of each feature vector.
    :ivar name: The name of this convolution.
    :ivar from_v: True if the message is passed from the variables to either the constraints or cut candidates.
    :ivar feature_module_left: A 64-node layer, which applies a weighted sum with bias to the input.
    :ivar feature_module_edge: A 64-node layer, which applies a weighted sum to the input.
    :ivar feature_module_right: A 64-node layer, which applies a weighted sum to the input.
    :ivar feature_module_final: A scaling operation and ReLU activation, followed by a 64-node layer which applies a
        weighted sum to the input.
    :ivar post_conv_module: A scaling operation.
    :ivar output_module: Two 64-node layers with ReLU activation.
    """

    def __init__(self, emb_size: int, name: str, from_v=False):
        super().__init__()
        self.emb_size = emb_size
        self.from_v = from_v

        # Feature modules (essentially weighted sums).
        self.feature_module_left = Sequential([
            Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer='orthogonal',
                  name=f'{name}_feat_left')])

        self.feature_module_edge = Sequential([
            Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer='orthogonal',
                  name=f'{name}_feat_edge')])

        self.feature_module_right = Sequential([
            Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer='orthogonal',
                  name=f'{name}_feat_right')])

        self.feature_module_final = Sequential([PreNormLayer(1, shift=False), Activation('relu'),
                                                Dense(units=self.emb_size, activation=None,
                                                      kernel_initializer='orthogonal', name=f'{name}_feat_final')])

        # Scaling operation.
        self.post_conv_module = Sequential([PreNormLayer(1, shift=False)])

        # Output layer.
        self.output_module = Sequential(
            [Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal', name=f'{name}_out_1'),
             Dense(units=self.emb_size, activation='relu', kernel_initializer='orthogonal', name=f'{name}_out_2')])

    def build(self, input_shapes):
        """Builds the model.

        Input is of the form [*l_shape*, *ei_shape*, *e_shape, *v_shape*], with the following parameters:

        - *l_shape*: The shape of the constraint or cut feature matrix.
        - *ei_shape*: The shape of the edge index matrix.
        - *e_shape*: The shape of the edge feature matrix.
        - *v_shape*: The shape of the variable feature matrix.

        :param input_shapes: The input shapes to use for building the model.
        """

        l_shape, ei_shape, e_shape, v_shape = input_shapes

        self.feature_module_left.build(l_shape)
        self.feature_module_edge.build(e_shape)
        self.feature_module_right.build(v_shape)
        self.feature_module_final.build([None, self.emb_size])
        self.post_conv_module.build([None, self.emb_size])
        self.output_module.build([None, self.emb_size + (l_shape[1] if self.from_v else v_shape[1])])
        self.built = True

    def call(self, inputs, training: bool):
        """Calls the model using the specified input, and performs a partial graph convolution.

        Input is of the form [*left_features*, *edge_indices*, *edge_features*, *variable_features*, *out_size*],
        with the following parameters:

        - *left_features*: A 2D constraint or cut feature tensor of shape (*n_left*, *n_features*).
        - *edge_indices*: A 2D edge index tensor of shape (2, *n_edges*).
        - *edge_features*: A 2D edge feature tensor of shape (*n_edges*, *n_edge_features*).
        - *variable_features*: A 2D variable feature tensor of shape (*n_variables*, *n_var_features*).
        - *out_size*: The size of the output (either *n_left* or *n_vars*).

        :param inputs: The convolution input.
        :param training: True if in training mode.
        :return: The convolution output, of shape (*n_left*, *emb_size*) if *self.from_v* is true, (*n_vars,
            *emb_size*) otherwise.
        """

        left_feats, edge_inds, edge_feats, var_feats, out_size = inputs

        if self.from_v:
            # The message-receiving side is the left-hand side of the bipartite graph.
            receiving_side = 0
            receiving_feats = left_feats
        else:
            # The message-receiving side is the right-hand side of the bipartite graph.
            receiving_side = 1
            receiving_feats = var_feats

        # Compute a joint feature representation for each edge.
        joint_features = self.feature_module_final(
            tf.gather(self.feature_module_left(left_feats), axis=0, indices=edge_inds[0]) + self.feature_module_edge(
                edge_feats) + tf.gather(self.feature_module_right(var_feats), axis=0, indices=edge_inds[1]))

        # Allocate the joint feature representation of an edge to the corresponding receiving node.
        conv_output = tf.scatter_nd(updates=joint_features, indices=tf.expand_dims(edge_inds[receiving_side], axis=1),
                                    shape=[out_size, self.emb_size])
        conv_output = self.post_conv_module(conv_output)

        # Concatenate and pass through the output module.
        output = self.output_module(tf.concat([conv_output, receiving_feats], axis=1))

        return output
