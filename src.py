import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from tqdm import tqdm

class Lake():

    def __init__(self, L, noise_level=1, wind_level=0.01, stepsize = 0.01):
        self.L = L
        self.noise_level = noise_level
        self.wind_level = wind_level
        self.stepsize = stepsize
        self.freqs = np.random.randint(1,10,self.noise_level) * np.pi / self.L
        self.amps = np.random.rand(self.noise_level) * self.wind_level
        self.x = np.arange(0, self.L+stepsize, stepsize)
    
    def profile(self, x):
        return jnp.sum(jnp.array([amp * jnp.sin(freq * x) for freq, amp in zip(self.freqs, self.amps)]))

    def vprofile(self, x):
        return jax.vmap(self.profile)(x)

    def slope(self, x):
        return jax.grad(self.profile)(x)

class Source():

    def __init__(self, x0, height, size, brightness_type='uniform'):
        self.x0 = x0
        self.height = height
        self.size = size
        self.brightness_type = brightness_type
        self.y = np.arange(0, self.height + 2 * self.size, 0.01)

    def brightness(self, y):
        if self.brightness_type == 'uniform':
            if np.abs(y - self.height) < self.size:
                return 1
            else:
                return 0
        elif self.brightness_type == 'gaussian':
            return jnp.exp(-(y-self.height)**2 / self.size**2)

class Ray():

    def __init__(self, x0, y0, slope):
        self.x0 = x0
        self.y0 = y0
        self.slope = slope

    def y(self, x):
        return self.slope * (x - self.x0) + self.y0

    def reflect(self,L):
        for x in np.flip(L.x):
            if self.y(x) < L.profile(x):
                slope_lake = L.slope(x)
                self.x0 = x
                self.y0 = L.profile(x)
                try:
                    self.slope = (-2 * slope_lake + self.slope - slope_lake ** 2 * self.slope) / (-1 + slope_lake ** 2 - 2 * slope_lake * self.slope)
                except ZeroDivisionError:
                    self.slope = self.slope
                break
        
    def brightness(self, S):
        self.brightness = S.brightness(self.y(S.x0))

class Screen():

    def __init__(self, x0, H, y_fake = -0.1, pixel=0.01):
        self.H = H
        self.x0 = x0
        self.pixel = pixel
        self.y_fake = y_fake

    def update(self, lake, source):
        print('Updating screen...')
        self.brightness = np.zeros_like(lake.x)
        for slope in tqdm(np.linspace(self.H/(self.x0 - lake.x[0]), self.H/(self.x0 - lake.x[-1]), len(lake.x))):
            ray = Ray(self.x0, self.H, slope)
            inds = np.where(ray.y(lake.x) < self.y_fake)[0]
            if len(inds) > 0:
                ray.reflect(lake)
                ray.brightness(source)
                self.brightness[inds[-1]] += ray.brightness
    
    def plot(self, source, lake):
        print('Plotting ...')
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.plot(lake.x, lake.vprofile(lake.x), color = 'cornflowerblue')
        max_brightness_source = np.max(source.brightness(source.y))
        max_brightness = np.max(self.brightness)
        [ax.plot(source.x0, source.y[i], 'x', color = 'red', alpha = float(source.brightness(source.y[i]) / max_brightness_source)) for i in range(len(source.y))]
        ax.vlines(self.x0, 0, self.H, color = 'black')
        [ax.plot(lake.x[i], self.y_fake, 'x', color = 'red', alpha = float(self.brightness[i] / max_brightness)) for i in range(len(lake.x))]
        ax.set_ylim([2.5 * self.y_fake, 1.5 * source.y[-1]])
        ax.set_axis_off()
        plt.show()