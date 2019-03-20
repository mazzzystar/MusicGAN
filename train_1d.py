import gc
from torch import optim
import json
import pickle
import datetime
from rnn import *
from util import *
from logger import *
from config import *
cuda = True if torch.cuda.is_available() else False

"""
############
# 1-d
############
* Only 1 z  or every z for the first discriminator, all later GRUCell input are the output of previous.
* Only 1 Discriminator for capture long-term dependency.(RF=131065, 8s)
"""

# =============Logger===============
LOGGER = logging.getLogger('MusicGAN')
LOGGER.setLevel(logging.DEBUG)

LOGGER.info('Initialized logger.')
init_console_logger(LOGGER)

# =============Parameters===============
one_z = False
same_z = False

args = parse_arguments()
epochs = args['num_epochs']
batch_size = args['batch_size']
latent_dim = args['latent_dim']
ngpus = args['ngpus']
model_size = args['model_size']
model_dir = make_path(os.path.join(args['output_dir'],
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
args['model_dir'] = model_dir
# save samples for every N epochs.
epochs_per_sample = args['epochs_per_sample']
# gradient penalty regularization factor.
lmbda = args['lmbda']

# Dir
audio_dir = args['audio_dir']
output_dir = args['output_dir']

# =============Network===============
rnnModel = RNN2(latent_dim, hidden_size=latent_dim, batch_size=batch_size)
hidden = rnnModel.init_hidden()
netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus)
# netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)
netD = WaveGANDiscriminator(model_size=model_size)

if cuda:
    rnnModel = rnnModel.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    # netG = torch.nn.DataParallel(netG).cuda()
    # netD = torch.nn.DataParallel(netD).cuda()

gen_params = list(rnnModel.parameters()) + list(netG.parameters())
# optimizerRNN = optim.Adam(rnnModel.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerG = optim.Adam(gen_params, lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerD = optim.Adam(netD.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))

# Sample noise used for generated output.
sample_noise = random_noise(latent_dim, same_z, one_z, cuda)

# Save config.
LOGGER.info('Saving configurations...')
config_path = os.path.join(model_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(args, f)

# Load data.
LOGGER.info('Loading audio data...')
audio_paths = gather_all_audios(audio_dir, max_len=MAX_DATA_NUM)
train_data, valid_data, test_data, train_size = split_data(audio_paths, args['valid_ratio'],
                                                           args['test_ratio'], batch_size)
TOTAL_TRAIN_SAMPLES = train_size
BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

train_iter = iter(train_data)
valid_iter = iter(valid_data)
test_iter = iter(test_data)

# =============Train===============
history = []
D_costs_train = []
D_wasses_train = []
D_costs_valid = []
D_wasses_valid = []
G_costs = []

alpha = 1.0

start = time.time()
LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))

for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D_cost_train_epoch = []
    D_wass_train_epoch = []
    D_cost_valid_epoch = []
    D_wass_valid_epoch = []
    G_cost_epoch = []
    for i in range(1, BATCH_NUM+1):
        one = torch.Tensor([1]).float()
        neg_one = one * -1
        if cuda:
            one = one.cuda()
            neg_one = neg_one.cuda()
        #############################
        # (1) Train Discriminator
        #############################
        rnnModel.zero_grad()
        netG.zero_grad()

        # Set Discriminator parameters to require gradients.
        for p in netD.parameters():
            p.requires_grad = True

        for iter_dis in range(5):
            netD.zero_grad()

            # Noise
            noise_D = random_noise(latent_dim, same_z, one_z, cuda)

            # a) compute loss contribution from real training data
            real_data = next(train_iter)['X']
            real_data_Var = numpy_to_var(real_data, cuda)

            D_real = netD(real_data_Var)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            gen_lst, hidden = pass_through_rnn_g(noise_D, hidden, same_z, one_z, rnnModel, netG, batch_size)
            fake = batch_fake_generator(gen_lst, batch_size)
            D_fake = netD(fake.detach())
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_data_Var.data,
                                                     fake.data, batch_size, lmbda,
                                                     use_cuda=cuda)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            #############################
            # (2) Compute Valid data
            #############################
            netD.zero_grad()

            valid_data_Var = numpy_to_var(next(valid_iter)['X'], cuda)
            D_real_valid = netD(valid_data_Var)
            D_real_valid = D_real_valid.mean()  # avg loss

            # b) compute loss contribution from generated data, then backprop.
            gen_lst, hidden = pass_through_rnn_g(noise_D, hidden, same_z, one_z, rnnModel, netG, batch_size)
            fake_valid = batch_fake_generator(gen_lst, batch_size)
            D_fake_valid = netD(fake_valid.detach())
            D_fake_valid = D_fake_valid.mean()

            # c) compute gradient penalty and backprop
            gradient_penalty_valid = calc_gradient_penalty(netD, valid_data_Var.data,
                                                           fake_valid.data, batch_size, lmbda,
                                                           use_cuda=cuda)
            # Compute metrics and record in batch history.
            D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
            D_wass_valid = D_real_valid - D_fake_valid

            if cuda:
                D_cost_train = D_cost_train.cpu()
                D_wass_train = D_wass_train.cpu()
                D_cost_valid = D_cost_valid.cpu()
                D_wass_valid = D_wass_valid.cpu()

            # Record costs
            D_cost_train_epoch.append(D_cost_train.data.numpy()[0])
            D_wass_train_epoch.append(D_wass_train.data.numpy()[0])
            D_cost_valid_epoch.append(D_cost_valid.data.numpy()[0])
            D_wass_valid_epoch.append(D_wass_valid.data.numpy()[0])

        #############################
        # (3) Train Generator & RNN
        #############################
        # Prevent discriminator update.
        for p in netD.parameters():
            p.requires_grad = False

        rnnModel.zero_grad()
        netG.zero_grad()
        netD.zero_grad()

        # Noise
        noise_G = random_noise(latent_dim, same_z, one_z, cuda)

        gen_lst, hidden = pass_through_rnn_g(noise_G, hidden, same_z, one_z, rnnModel, netG, batch_size)
        fake = batch_fake_generator(gen_lst, batch_size)
        G = netD(fake)
        G = G.mean()

        # Update gradients.
        G.backward(neg_one)
        optimizerG.step()

        """
        Here detach the hidden to disconnect the graph between batches.
        See https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time-while-using-custom-loss-function/12360/2
        """
        hidden = hidden.detach()

        G_cost = -G
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy()[0])

        # clean
        gc.collect()

        # (BATCH_NUM // 5)
        LOG_PER = (BATCH_NUM // 5) if MAX_DATA_NUM > 1000 else 1
        if i % LOG_PER == 0:
            LOGGER.info("{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                             i, BATCH_NUM,
                                                                                             D_cost_train.data.numpy()[0],
                                                                                             D_wass_train.data.numpy()[0],
                                                                                             G_cost.data.numpy()[0]))

    # Save the average cost of batches in every epoch.
    D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
    D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
    D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
    D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs_train.append(D_cost_train_epoch_avg)
    D_wasses_train.append(D_wass_train_epoch_avg)
    D_costs_valid.append(D_cost_valid_epoch_avg)
    D_wasses_valid.append(D_wass_valid_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} "
                "| G_cost:{:.4f}".format(time_since(start),
                                         D_cost_train_epoch_avg,
                                         D_wass_train_epoch_avg,
                                         D_cost_valid_epoch_avg,
                                         D_wass_valid_epoch_avg,
                                         G_cost_epoch_avg))

    # Generate audio samples.
    if epoch % epochs_per_sample == 0:
        LOGGER.info("Generating samples...")
        gen_lst, hidden = pass_through_rnn_g(sample_noise, hidden, same_z, one_z, rnnModel, netG, batch_size)
        sample_out = batch_fake_generator(gen_lst, batch_size)
        # sample_out = netG(sample_noise_Var)
        if cuda:
            sample_out = sample_out.cpu()
        sample_out = sample_out.data.numpy()[:args['sample_size'], :, :]
        save_samples(sample_out, epoch, output_dir)
        hidden = hidden.detach()

    # TODO
    # Early stopping by Inception Score(IS)

LOGGER.info('>>>>>>>Training finished !<<<<<<<')

# Save model
LOGGER.info("Saving models...")
netD_path = os.path.join(output_dir, "discriminator.pkl")
netG_path = os.path.join(output_dir, "generator.pkl")
netRNN_path = os.path.join(output_dir, "rnn.pkl")
torch.save(netD.state_dict(), netD_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
torch.save(netG.state_dict(), netG_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
torch.save(rnnModel.state_dict(), netRNN_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

# Plot loss curve.
LOGGER.info("Saving loss curve...")
# SKIP_FIRST_EPOCHS = 10
plot_loss(D_costs_train, D_wasses_train,
          D_costs_valid, D_wasses_valid, G_costs, output_dir, name='1z1dist')
plot_loss_wass(D_wasses_train, D_wasses_valid, G_costs, output_dir, name='1z1dist_wass')

LOGGER.info("All finished!")



