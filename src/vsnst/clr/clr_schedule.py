import tensorflow as tf


class CLRSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, cycle_length, total_iterations=None, div_factor=10.0, epoch_decay=0.5
    ):
        self._warmup_end_step = tf.cast(cycle_length * 0.1, dtype=tf.int64)
        self._max_end_step = self._warmup_end_step + tf.cast(
            cycle_length * 0.4, dtype=tf.int64
        )
        self._decay_end_step = self._max_end_step + tf.cast(
            cycle_length * 0.5, dtype=tf.int64
        )
        self.div_factor = tf.cast(div_factor, dtype=tf.float32)
        self.total_iterations = total_iterations
        self.epoch_decay = epoch_decay
        if total_iterations is not None:
            self.total_iterations = tf.cast(total_iterations, dtype=tf.int64)

    def __call__(self, step):
        epoch = None
        max_factor = 1.0
        min_factor = 1 / self.div_factor
        if self.total_iterations is not None:
            epoch = (step - 1) // self.total_iterations
            min_factor = min_factor * (
                self.epoch_decay ** tf.cast(epoch, dtype=tf.float32)
            )
            max_factor = min_factor * self.div_factor
            step = step - self.total_iterations * epoch

        def warmup():
            # interpolate between initial and max lr
            lr_factor = min_factor + (max_factor - min_factor) * tf.cast(
                step / self._warmup_end_step, dtype=tf.float32
            )
            return tf.cast(lr_factor, dtype=tf.float32)

        def max_lr():
            # remain at max for a period
            return tf.cast(max_factor, dtype=tf.float32)

        def initial_decay():
            # decay at half the speed we warmed up at
            decay_step = step - self._max_end_step
            lr_factor = max_factor - (max_factor - min_factor) * tf.cast(
                decay_step / (self._decay_end_step - self._max_end_step),
                dtype=tf.float32,
            )
            return tf.cast(lr_factor, dtype=tf.float32)

        def final_decay():
            # then exponential decay from there
            final_step = tf.cast(step - self._decay_end_step, dtype=tf.float32)
            base = tf.cast(1 - 1 / self._decay_end_step, dtype=tf.float32)
            lr_factor = min_factor * tf.math.pow(base, final_step)
            return lr_factor

        def default_lr():
            return tf.cast(1 / self.div_factor, dtype=tf.float32)

        learning_rate = tf.case(
            [
                (tf.less_equal(step, self._warmup_end_step), warmup),
                (tf.less_equal(step, self._max_end_step), max_lr),
                (tf.less_equal(step, self._decay_end_step), initial_decay),
            ],
            # (tf.less_equal(step, self.total_iterations), final_decay)],
            default=final_decay,
        )

        return learning_rate

    def get_momentum(self, step):
        if self.total_iterations is not None:
            step = step - self.total_iterations * ((step - 1) // self.total_iterations)

        def warmup():
            # interpolate between initial and max momentum
            momentum = 0.95 - 0.1 * (step / self._warmup_end_step)
            return tf.cast(momentum, dtype=tf.float32)

        def max_lr():
            # remain at min momentum while learning rate is maxed
            return tf.cast(0.85, dtype=tf.float32)

        def initial_decay():
            # decay back up to max momentum
            decay_step = step - self._max_end_step
            momentum = 0.85 + 0.1 * (
                decay_step / (self._decay_end_step - self._max_end_step)
            )
            return tf.cast(momentum, dtype=tf.float32)

        def final_decay():
            # remain at highest momentum
            return tf.cast(0.95, dtype=tf.float32)

        learning_rate = tf.case(
            [
                (tf.less_equal(step, self._warmup_end_step), warmup),
                (tf.less_equal(step, self._max_end_step), max_lr),
                (tf.less_equal(step, self._decay_end_step), initial_decay),
            ],
            default=final_decay,
        )

        return learning_rate
