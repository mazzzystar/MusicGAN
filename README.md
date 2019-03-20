# MusicGAN
This was a previous work I did at NAVER. The motivation for this project is to **generate long-term dependency raw piano audio**. We use RNN combined with [WaveGAN](https://arxiv.org/abs/1802.04208) to solve the problem of "long-term dependency", and the results is: https://soundcloud.com/mazzzystar/sets/only-1-discriminator-to-control-both-local-long-term

## Architecture
The archicture is as below:
![](pic/architecture.png)


## TODO
- [ ] Try to improve the implementation of current `WaveGAN` to get better quality of audio.
- [ ] Combine MusicGAN with VQVAE.