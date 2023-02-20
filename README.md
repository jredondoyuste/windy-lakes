# windy-lakes

Have you ever walked by the lakes of Copenhagen on a windy night?

If you have, for sure you have noticed the distorted image of the buildings of Norrebro in the surface of the lake. It is not an effect of Snaps, it is just physics! In fact, this is due to the waves of the water caused by the eternal wind in the city. 

In order to test this hypothesis, I spent an unnecesary amount of time writing this simple ray-tracing code! 

### Requirements

The code makes use of the following packages (readily available via pip): 

* numpy
* matplotlib
* tqdm
* jax

### Usage

The fundaments are the following classes: 

* Lake: creates the lake object, with custom wave heights and noise levels. 
* Source: so far limited options, it creates an extended source of light.
* Ray: no need to use this one, useful to compute the reflections.
* Screen: the position of the observer. 

Then using it only takes a handful of lines, see Example.ipynb above! 

If you want to support my procrastination exercises, please consider starring this repo :) 
