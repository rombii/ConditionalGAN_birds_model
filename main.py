import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

from Model import generator as gen
from Model import discriminator as disc
from Model import trainer as train

from Data import loader as data_loader


data = data_loader.load_dataset()

gen_model = gen.build()

disc_model = disc.build()


epochs = 1000

gen_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.0001,
    end_learning_rate=0.0,
    decay_steps=len(data) * epochs,
)

desc_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.0001,
    end_learning_rate=0.0,
    decay_steps=len(data) * epochs,
)

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_schedule, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=desc_schedule, beta_1=0.5)

train.train(data, epochs,
            gen_model, disc_model,
            gen_optimizer, disc_optimizer)







