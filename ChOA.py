import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.optimizers.legacy import optimizer_v2


class ChimpOptimizer(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        name="RMSprop",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)

        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError(
                "`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)})."
            )
        self._set_hyper("momentum", momentum)

        self.epsilon = epsilon or backend_config.epsilon()
        self.centered = centered

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "mg")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        rho = tf.identity(self._get_hyper("rho", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                rho=rho,
                momentum=tf.identity(self._get_hyper("momentum", var_dtype)),
                one_minus_rho=1.0 - rho,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return tf.raw_ops.ResourceApplyCenteredRMSProp(
                    var=var.handle,
                    mg=mg.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    use_locking=self._use_locking,
                )
            else:
                return tf.raw_ops.ResourceApplyRMSProp(
                    var=var.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    use_locking=self._use_locking,
                )
        else:
            rms_t = coefficients["rho"] * rms + coefficients[
                "one_minus_rho"
            ] * tf.square(grad)
            rms_t = tf.compat.v1.assign(rms, rms_t, use_locking=self._use_locking)
            denom_t = rms_t
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
                mg_t = tf.compat.v1.assign(mg, mg_t, use_locking=self._use_locking)
                denom_t = rms_t - tf.square(mg_t)
            var_t = var - coefficients["lr_t"] * grad / (
                tf.sqrt(denom_t) + coefficients["epsilon"]
            )
            return tf.compat.v1.assign(var, var_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return tf.raw_ops.ResourceSparseApplyCenteredRMSProp(
                    var=var.handle,
                    mg=mg.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    indices=indices,
                    use_locking=self._use_locking,
                )
            else:
                return tf.raw_ops.ResourceSparseApplyRMSProp(
                    var=var.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    indices=indices,
                    use_locking=self._use_locking,
                )
        else:
            rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
            rms_t = tf.compat.v1.assign(
                rms, rms * coefficients["rho"], use_locking=self._use_locking
            )
            with tf.control_dependencies([rms_t]):
                rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
                rms_slice = tf.gather(rms_t, indices)
            denom_slice = rms_slice
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg_scaled_g_values = grad * coefficients["one_minus_rho"]
                mg_t = tf.compat.v1.assign(
                    mg, mg * coefficients["rho"], use_locking=self._use_locking
                )
                with tf.control_dependencies([mg_t]):
                    mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
                    mg_slice = tf.gather(mg_t, indices)
                    denom_slice = rms_slice - tf.square(mg_slice)
            var_update = self._resource_scatter_add(
                var,
                indices,
                coefficients["neg_lr_t"]
                * grad
                / (tf.sqrt(denom_slice) + coefficients["epsilon"]),
            )
            if self.centered:
                return tf.group(*[var_update, rms_t, mg_t])
            return tf.group(*[var_update, rms_t])

    def set_weights(self, weights):
        params = self.weights
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._initial_decay,
                "rho": self._serialize_hyperparameter("rho"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config
