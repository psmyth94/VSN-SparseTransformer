import tensorflow as tf

from .clr_schedule import CLRSchedule


class CLRAdam(tf.keras.optimizers.Adam):
    def __init__(
        self,
        learning_rate,
        cycle_length,
        total_iterations=None,
        div_factor=10.0,
        epoch_decay=0.5,
    ):
        self.one_cycle_schedule = CLRSchedule(
            cycle_length, total_iterations, div_factor, epoch_decay
        )
        super(CLRAdam, self).__init__(
            learning_rate=lambda: learning_rate
            * self.one_cycle_schedule(self.iterations),
            beta_1=lambda: self.one_cycle_schedule.get_momentum(self.iterations),
            beta_2=0.99,
        )


