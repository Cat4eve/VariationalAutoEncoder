import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cv2 import imread, resize
import pandas as pd
import subprocess


class VAE(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.dimZ = 100
        print('a')
        self.data, self.attrs = self.fetch_lfw_dataset()
        print('a')
        self.X_train = self.data[:10000].reshape((10000, -1))
        print(self.X_train.shape)
        self.X_val = self.data[10000:].reshape((-1, self.X_train.shape[1]))
        print(self.X_val.shape)

        self.image_h = self.data.shape[1]
        self.image_w = self.data.shape[2]

        self.X_train = np.float32(self.X_train)
        self.X_train = self.X_train/255
        self.X_val = np.float32(self.X_val)
        self.X_val = self.X_val/255

        self.encoder = tf.keras.Model(
            name='encoder',
            inputs=[
                tf.keras.layers.Dense(self.X_train.shape[1]),
                tf.keras.layers.Dense(self.dimZ * 4)
            ],
            outputs=[
                tf.keras.layers.Dense(self.dimZ * 2)
            ]
        )
        self.decoder = tf.keras.Model(
            name='decoder',
            inputs=[
                tf.keras.layers.Dense(self.dimZ),
                tf.keras.layers.Dense(self.dimZ * 4)
            ],
            outputs=[
                tf.keras.layers.Dense(self.X_train.shape[1])
            ]
        )

    @tf.function
    def KL_divergence(self, mu, sigma):
        return -0.5 * np.sum(1 + np.log(sigma**2) - mu**2 - sigma**2)

    @tf.function
    def log_likelihood(self, x, z):
        return np.sum(-(((np.abs(x-z))**2)/2))

    def train_step(self, data, true):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            mu = [], sigma = []
            for i in range(self.dimZ):
                mu.append(z[i])
                sigma.append(z[i+self.dimZ])
            loss = self.KL_divergence(np.array(mu), np.array(sigma)) + np.sum((self.forward(data) - true)**2, axis=0)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))



    def call(self, inputs, training=False):
        if not training: return self.forward(inputs)

    def forward(self, inputs):
        z = self.encoder(inputs)
        y = []
        for i in range(self.dimZ):
            y.append(np.random.normal(z[i], z[i+self.dimZ], self.dimZ))
        return self.decoder(np.array(y))

    def fetch_lfw_dataset(self, attrs_name="lfw_attributes.txt",
                      images_name="lfw-deepfunneled",
                      raw_images_name="lfw",
                      use_raw=False,
                      dx=80, dy=80,
                      dimx=45, dimy=45
                      ):
    # download if not exists
        if (not use_raw) and not os.path.exists(images_name):
            print("images not found, downloading...")
            subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -OutFile tmp.tgz"], shell=True)
            print("extracting...")
            subprocess.run(["tar", "xvzf", "tmp.tgz"], shell=True)
            os.remove("tmp.tgz")
            print("done")
            assert os.path.exists(os.path.join(images_name, "lfw"))

        if use_raw and not os.path.exists(raw_images_name):
            print("images not found, downloading...")
            subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw.tgz -OutFile tmp.tgz"], shell=True)
            subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -OutFile tmp.tgz"], shell=True)
            print("extracting...")
            subprocess.run(["tar", "xvzf", "tmp.tgz"], shell=True)
            os.remove("tmp.tgz")
            print("done")
            assert os.path.exists(os.path.join(raw_images_name, "lfw"))

        if not os.path.exists(attrs_name):
            print("attributes not found, downloading...")
            subprocess.run(["powershell", "Invoke-WebRequest -Uri http://www.cs.columbia.edu/CAVE/databases/pubfig/download/%s -OutFile %s" % (attrs_name, attrs_name)])
            print("done")

        # read attrs
        df_attrs = pd.read_csv(attrs_name, sep='\t', skiprows=1,)
        df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
        df_attrs.imagenum = df_attrs.imagenum.astype(np.int64)

        # read photos
        dirname = raw_images_name if use_raw else images_name
        photo_ids = []
        for dirpath, dirnames, filenames in os.walk(dirname):
            for fname in filenames:
                if fname.endswith(".jpg"):
                    fpath = os.path.join(dirpath, fname)
                    photo_id = fname[:-4].replace('_', ' ').split()
                    person_id = ' '.join(photo_id[:-1])
                    photo_number = int(photo_id[-1])
                    photo_ids.append({'person': person_id, 'imagenum': photo_number, 'photo_path': fpath})

        photo_ids = pd.DataFrame(photo_ids)

        # mass-merge
        # (photos now have same order as attributes)
        df = pd.merge(df_attrs, photo_ids, on=('person', 'imagenum'))

        assert len(df) == len(df_attrs), "lost some data when merging dataframes"

        # image preprocessing
        all_photos = df['photo_path'].apply(imread) \
            .apply(lambda img: img[dy:-dy, dx:-dx]) \
            .apply(lambda img: resize(img, (dimx, dimy)))

        all_photos = np.stack(all_photos.values).astype('uint8')
        all_attrs = df.drop(["photo_path", "person", "imagenum"], axis=1)

        return all_photos, all_attrs




vae = VAE()