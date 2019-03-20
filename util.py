import os
import time
import math
import torch
import glob
import pickle
import random
import logging
import librosa
import argparse
import pescador
import numpy as np
from torch import autograd
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from config import *
LOGGER = logging.getLogger('MusicGAN')
LOGGER.setLevel(logging.DEBUG)


def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Adapted from @jtcramer https://github.com/jtcramer/wavegan/blob/master/sample.py.
def sample_generator(filepath, window_length=int(16384 * SECOND * G_NUMS), fs=16000):
    """
    Audio sample generator
    """
    try:
        # print(">>>>>>>" + filepath)
        audio_data, _ = librosa.load(filepath, sr=fs)

        # Clip magnitude
        max_mag = np.max(np.abs(audio_data))
        if max_mag > 1:
            audio_data /= max_mag
    except Exception as e:
        LOGGER.error("Could not load {}: {}".format(filepath, str(e)))
        raise StopIteration

    # Pad audio to >= window_length.
    audio_len = len(audio_data)
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        audio_len = len(audio_data)

    while True:
        if audio_len == window_length:
            # If we only have a single 1*window_length audio, just yield.
            sample = audio_data
        else:
            # Sample a random window from the audio
            start_idx = np.random.randint(0, audio_len - window_length)
            end_idx = start_idx + window_length
            sample = audio_data[start_idx:end_idx]

        sample = sample.astype('float32')
        assert not np.any(np.isnan(sample))

        yield {'X': sample}


def find_files(path): return glob.glob(path)


def gather_all_audios(datapath, max_len):
    """
    Only collect Mozart's audio.
    """
    all_audios = []
    count = 0
    for filename in find_files(datapath + '/*/*.wav'):
        last_name = filename.split('/')[-1]
        # print(last_name)
        if 'mozart' not in last_name.lower():
            continue
        if count > max_len:
            break
        all_audios.append(filename)
        count += 1
    for filename in find_files(datapath + '/*/*.mp3'):
        last_name = filename.split('/')[-1]
        if 'mozart' not in last_name.lower():
            continue
        if count > max_len:
            break
        all_audios.append(filename)
        count += 1
    return all_audios


# def get_all_audio_filepaths(audio_dir):
#     return [os.path.join(root, fname)
#             for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
#             for fname in file_names
#             if (fname.lower().startswith('zkocsismozart') and fname.lower().endswith('.wav'))]


def batch_generator(audio_path_list, batch_size):
    streamers = []
    for audio_path in audio_path_list:
        s = pescador.Streamer(sample_generator, audio_path)
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen


def split_data(audio_path_list, valid_ratio, test_ratio, batch_size):
    num_files = len(audio_path_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    assert num_valid > 0 and num_test > 0 and num_train > 0

    # Random shuffle the audio_path_list for splitting.
    random.shuffle(audio_path_list)

    valid_files = audio_path_list[:num_valid]
    test_files = audio_path_list[num_valid:num_valid + num_test]
    train_files = audio_path_list[num_valid + num_test:]
    train_size = len(train_files)

    train_data = batch_generator(train_files, batch_size)
    valid_data = batch_generator(valid_files, batch_size)
    test_data = batch_generator(test_files, batch_size)

    return train_data, valid_data, test_data, train_size


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, use_cuda=False):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # print("alpha_size={}".format(alpha.shape))
    # print("real_data_size={}".format(real_data.shape))
    # print("fake_data_size={}".format(fake_data.shape))

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def numpy_to_var(numpy_data, cuda):
    """
    Convert numpy array to Variable.
    """
    data = numpy_data[:, np.newaxis, :]
    data = torch.Tensor(data)
    if cuda:
        data = data.cuda()
    return Variable(data, requires_grad=False)


def numpy_to_var_piece(numpy_data, cuda):
    """
    Random crop a piece of music length=1*16384
    """
    batch_size, audio_len = numpy_data.shape[0], numpy_data.shape[1]

    length = int(16384 * 1 * SECOND)
    start = random.randint(0, audio_len - length)
    data_seg = numpy_data[:, start:start + length].copy()
    data_tensor = torch.Tensor(data_seg)
    data_tensor = data_tensor.view(batch_size, 1, -1)
    if cuda:
        data_tensor = data_tensor.cuda()
    return Variable(data_tensor, requires_grad=False)


traindata = DATASET_NAME
output = make_path('output/')


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a WaveGAN on a given set of audio')

    parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=64,
                        help='Model size parameter used in WaveGAN')
    parser.add_argument('-pssf', '--phase-shuffle-shift-factor', dest='shift_factor', type=int, default=2,
                        help='Maximum shift used by phase shuffle')
    parser.add_argument('-psb', '--phase-shuffle-batchwise', dest='batch_shuffle', action='store_true',
                        help='If true, apply phase shuffle to entire batches rather than individual samples')
    parser.add_argument('-ppfl', '--post-proc-filt-len', dest='post_proc_filt_len', type=int, default=512,
                        help='Length of post processing filter used by generator. Set to 0 to disable.')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2,
                        help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-vr', '--valid-ratio', dest='valid_ratio', type=float, default=0.1,
                        help='Ratio of audio files used for validation')
    parser.add_argument('-tr', '--test-ratio', dest='test_ratio', type=float, default=0.1,
                        help='Ratio of audio files used for testing')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size used for training')
    parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('-ld', '--latent-dim', dest='latent_dim', type=int, default=128,
                        help='Size of latent dimension used by generator')
    parser.add_argument('-eps', '--epochs-per-sample', dest='epochs_per_sample', type=int, default=SAMPLE_EVERY,
                        help='How many epochs between every set of samples generated for inspection')
    parser.add_argument('-ss', '--sample-size', dest='sample_size', type=int, default=10,
                        help='Number of inspection samples generated')
    parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=10.0,
                        help='Gradient penalty regularization factor')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=1e-4,
                        help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-audio_dir', '--audio_dir', dest='audio_dir', type=str, default=traindata, help='Path to directory containing audio files')
    parser.add_argument('-output_dir', '--output_dir', dest='output_dir', type=str, default=output, help='Path to directory where model files will be output')
    args = parser.parse_args()
    return vars(args)


def pass_through_rnn_g(z, hidden, same_z, one_z, rnnModel, netG, batch_size):
    """
    First make every input pass through rnn and generator

    z: (batch, input_size * G_NUMS)
    """
    if one_z:
        z = z.view(batch_size, 1, -1)
    elif same_z:
        z = z.view(batch_size, 1, -1)
        z = z.expand(batch_size, G_NUMS, z.shape[-1])
    else:
        z = z.view(batch_size, G_NUMS, -1)
    outputs, hidden = rnnModel(z, hidden)
    print(outputs[0].shape)

    if one_z:
        output_lst = [outputs[i] for i in range(G_NUMS)]
    else:
        output_lst = [outputs[i].transpose(0, 1) for i in range(G_NUMS)]
    print(output_lst[0].shape)
    gen_lst = [netG(output_lst[i]) for i in range(G_NUMS)]
    return gen_lst, hidden


def batch_fake_generator(gen_lst, batch_size):
    """
    output size: (batch_size, 1, 16384 * G_NUMS)
    """
    gen_cat = torch.cat(gen_lst, dim=-1)
    gen_cat = gen_cat.view(batch_size, 1, -1)
    return gen_cat


def piece_fake_generator(gen_lst, batch_size):
    """
    output size: (batch_size, 1, 16384)
    """
    # random choose 1 output from 0-G_NUMS, so the
    # output from (batch_size, G_NUMS, 16384) -> (batch_size, 1, 16384)
    output_lst = []
    for i in range(batch_size):
        j = random.randint(0, G_NUMS-1)
        cur_out = gen_lst[j]
        index = random.randint(0, batch_size-1)
        output_lst.append(cur_out[index].view(1, 1, -1))

    out_cat = torch.cat(output_lst, dim=0)  # ->(batch_size, 1, 16384)
    # print("output_cat shape={}".format(out_cat.shape))
    return out_cat


def random_noise(latent_dim, same_z, one_z, cuda):
    if same_z or one_z:
        noise = torch.Tensor(BATCH_SIZE, latent_dim).uniform_(-1, 1)
    else:
        noise = torch.Tensor(BATCH_SIZE, latent_dim * G_NUMS).uniform_(-1, 1)
    if cuda:
        noise = noise.cuda()
    noise_var = Variable(noise, requires_grad=False)
    return noise_var


#####################
# Save output & model
#####################
def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Save output samples.
    """
    sample_dir = make_path(os.path.join(output_dir, str(epoch)))
    print("epoch_samples_shape={}".format(epoch_samples.shape))

    for idx, sample in enumerate(epoch_samples):
        # print("sample_shape={}".format(sample.shape))
        output_path = os.path.join(sample_dir, "{}.wav".format(idx+1))
        sample = sample[0]
        librosa.output.write_wav(output_path, sample, fs)


def save_model(epoch, output_dir, netD1, netD2, netG, rnnModel):
    # Save model
    LOGGER.info("Saving models...")
    netD1_path = os.path.join(output_dir, "smallDiscriminator" + str(epoch) + ".pkl")
    netD2_path = os.path.join(output_dir, "bigDiscriminator" + str(epoch) + ".pkl")
    netG_path = os.path.join(output_dir, "generator" + str(epoch) + ".pkl")
    netRNN_path = os.path.join(output_dir, "rnn" + str(epoch) + ".pkl")
    torch.save(netD1.state_dict(), netD1_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(netD2.state_dict(), netD2_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(netG.state_dict(), netG_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(rnnModel.state_dict(), netRNN_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


########
# Plot
########
def plot_loss(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid,
              G_cost, save_path, name):
    assert len(D_cost_train) == len(D_wass_train) == len(D_cost_valid) == len(D_wass_valid) == len(G_cost)

    epoch = len(D_cost_train)
    curve_name = "curve_" + str(name) + "_" + str(epoch) + ".png"
    save_path = os.path.join(save_path, curve_name)

    x = range(len(D_cost_train))

    y1 = D_cost_train
    y2 = D_wass_train
    y3 = D_cost_valid
    y4 = D_wass_valid
    y5 = G_cost

    fig = plt.figure()
    fig.clf()
    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='D_wass_train')
    plt.plot(x, y3, label='D_loss_valid')
    plt.plot(x, y4, label='D_wass_valid')
    plt.plot(x, y5, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)


def plot_loss_wass(D_wass_train, D_wass_valid, G_cost, save_path, name):
    assert len(D_wass_train) == len(D_wass_valid) == len(G_cost)

    epoch = len(D_wass_train)
    curve_name = "curve_" + str(name) + "_" + str(epoch) + ".png"
    save_path = os.path.join(save_path, curve_name)

    x = range(len(D_wass_train))

    y1 = D_wass_train
    y2 = D_wass_valid
    y3 = G_cost

    fig = plt.figure()
    fig.clf()
    plt.plot(x, y1, label='D_wass_train')
    plt.plot(x, y2, label='D_wass_valid')
    plt.plot(x, y3, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)