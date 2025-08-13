
---

## **💻 Example Code Skeleton**

### `models/vae.py`
```python
import tensorflow as tf
from tensorflow.keras import layers

def build_encoder(latent_dim):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name="encoder")
