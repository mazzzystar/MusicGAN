import gc
from torch import optim
import json
import datetime
from rnn import *
from util import *
from logger import *
from config import *
cuda = True if torch.cuda.is_available() else False

"""
############
# 2-d
############
* Only 1 z  or every z for the first discriminator, all later GRUCell input are the output of previous.
* D1(RF=8185, 0.5s) & D2 (RF=131065, 8s) for both local texture & long-term dependency.
"""

# =============Logger===============
LOGGER = logging.getLogger('muscGAN')
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
if one_z:
    rnnModel = RNN2(latent_dim, hidden_size=latent_dim, batch_size=batch_size)
else:
    rnnModel = RNN(latent_dim, hidden_size=latent_dim, batch_size=batch_size)
hidden = rnnModel.init_hidden()
netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, upsample=True)
# netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)

# netD1 in charge of training G, netD2 in charge of training RNN.

# netD1: Receptive Field(RF) = 8185
netD1 = WaveGANDiscriminator(model_size=model_size, name='d1')
# netD2: Receptive Field = 86956
netD2 = WaveGANDiscriminator(model_size=model_size, name='d2')

if cuda:
    rnnModel = rnnModel.cuda()
    netG = torch.nn.DataParallel(netG).cuda()
    netD1 = torch.nn.DataParallel(netD1).cuda()
    netD2 = torch.nn.DataParallel(netD2).cuda()

# gen_params = list(rnnModel.parameters()) + list(netG.parameters())
optimizerRNN = optim.Adam(rnnModel.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerG = optim.Adam(netG.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerD1 = optim.Adam(netD1.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerD2 = optim.Adam(netD2.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))

# ========= Initialization & data preparation.
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

# ===================Train=================
history = []

D1_costs_train = []
D1_wasses_train = []
D1_costs_valid = []
D1_wasses_valid = []

D2_costs_train = []
D2_wasses_train = []
D2_costs_valid = []
D2_wasses_valid = []

G_costs = []
G_whole_costs = []

start = time.time()
LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))

for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D1_cost_train_epoch = []
    D1_wass_train_epoch = []
    D1_cost_valid_epoch = []
    D1_wass_valid_epoch = []

    D2_cost_train_epoch = []
    D2_wass_train_epoch = []
    D2_cost_valid_epoch = []
    D2_wass_valid_epoch = []

    G_cost_epoch = []
    G_whole_epoch = []

    for i in range(1, BATCH_NUM+1):
        one = torch.Tensor([1]).float()
        neg_one = one * -1
        if cuda:
            one = one.cuda()
            neg_one = neg_one.cuda()

        """
        We train G and RNN separately, train <G, D1> with
        fixed RNN, train <RNN, D2> with fixed G.
        """

        #######################################
        # ========== Train D1 & RNN+G ============
        #######################################
        netD1.zero_grad()
        netG.zero_grad()
        rnnModel.zero_grad()

        for p in netD1.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = True
        for p in rnnModel.parameters():
            p.requires_grad = True
        for p in netD2.parameters():
            p.requires_grad = False

        train_real_data = next(train_iter)['X']

        noise_D = random_noise(latent_dim, same_z, one_z, cuda)
        gen_lst, hidden = pass_through_rnn_g(noise_D, hidden, same_z, one_z, rnnModel, netG, batch_size)
        fake_piece = piece_fake_generator(gen_lst, batch_size)

        # ========== Train D1 ============
        for index in range(G_NUMS):
            netD1.zero_grad()

            train_real_piece_Var = numpy_to_var_piece(train_real_data, cuda)
            fake_i = gen_lst[index].view(batch_size, 1, -1)
            # print("D1 fake data shape:{}".format(fake_i.shape))

            D1_real = netD1(train_real_piece_Var)
            D1_real = D1_real.mean()  # avg loss
            # print("D1_real={}".format(D1_real.cpu().data.numpy()[0]))
            D1_real.backward(neg_one)  # loss * -1

            # print("D1 fake data shape:{}".format(fake.shape))
            D1_fake = netD1(fake_i.detach())
            D1_fake = D1_fake.mean()
            # print("D1_fake={}".format(D1_fake.cpu().data.numpy()[0]))
            D1_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty1 = calc_gradient_penalty(netD1, train_real_piece_Var.data,
                                                      fake_i.data, batch_size, lmbda,
                                                      use_cuda=cuda)
            # print("gradient_penalty1={}".format(gradient_penalty1.cpu().data.numpy()[0]))
            gradient_penalty1.backward(one)

            # Compute metrics and record in batch history.
            D1_cost_train = D1_real - D1_fake + gradient_penalty1
            D1_wass_train = D1_real - D1_fake

            if cuda:
                D1_cost_train = D1_cost_train.cpu()
                D1_wass_train = D1_wass_train.cpu()

            D1_cost_train_epoch.append(D1_cost_train.data.numpy()[0])
            D1_wass_train_epoch.append(D1_wass_train.data.numpy()[0])

            # Update gradient of discriminator D1.
            optimizerD1.step()
            optimizerD1.zero_grad()

        # ========== Train RNN+G ============
        for p in netD1.parameters():
            p.requires_grad = False

        netD1.zero_grad()
        netG.zero_grad()
        rnnModel.zero_grad()

        G = netD1(fake_piece)
        G = G.mean()

        # Update gradients.
        G.backward(neg_one)
        optimizerG.step()
        optimizerG.zero_grad()
        optimizerRNN.step()
        optimizerRNN.zero_grad()

        G_cost = -G
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy()[0])

        hidden = hidden.detach()

        ########################################
        # ==========Train D2 & RNN+G =============
        ########################################
        for p in netD1.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True
        for p in netD2.parameters():
            p.requires_grad = True
        for p in rnnModel.parameters():
            p.requires_grad = True

        netD2.zero_grad()
        netG.zero_grad()
        rnnModel.zero_grad()

        # ==========Train D2 =============
        for iter_dis in range(5):
            netD2.zero_grad()

            noise_D = random_noise(latent_dim, same_z, one_z, cuda)
            gen_lst, hidden = pass_through_rnn_g(noise_D, hidden, same_z, one_z, rnnModel, netG, batch_size)
            fake = batch_fake_generator(gen_lst, batch_size)

            train_real_data = next(train_iter)['X']

            # a) compute loss contribution from real training data
            train_real_Var = numpy_to_var(train_real_data, cuda)
            # print("D2 real data shape:{}".format(train_real_Var.shape))

            D2_real = netD2(train_real_Var)
            D2_real = D2_real.mean()  # avg loss
            # print("D2_real={}".format(D2_real.data.numpy()[0]))
            D2_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            # print("D2 fake data shape:{}".format(fake.shape))
            D2_fake = netD2(fake.detach())
            D2_fake = D2_fake.mean()
            # print("D2_fake={}".format(D2_fake.cpu().data.numpy()[0]))
            D2_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty2 = calc_gradient_penalty(netD2, train_real_Var.data,
                                                      fake.data, batch_size, lmbda,
                                                      use_cuda=cuda)
            gradient_penalty2.backward(one)

            # Compute metrics and record in batch history.
            D2_cost_train = D2_real - D2_fake + gradient_penalty2
            D2_wass_train = D2_real - D2_fake

            if cuda:
                D2_cost_train = D2_cost_train.cpu()
                D2_wass_train = D2_wass_train.cpu()

            D2_cost_train_epoch.append(D2_cost_train.data.numpy()[0])
            D2_wass_train_epoch.append(D2_wass_train.data.numpy()[0])

            # Update gradient of discriminator D2.
            optimizerD2.step()
            optimizerD2.zero_grad()

        # ==========Train G & RNN =============
        for p in netD2.parameters():
            p.requires_grad = False

        netD2.zero_grad()
        netG.zero_grad()
        rnnModel.zero_grad()

        noise_G = random_noise(latent_dim, same_z, one_z, cuda)
        gen_lst, hidden = pass_through_rnn_g(noise_G, hidden, same_z, one_z, rnnModel, netG, batch_size)
        fake = batch_fake_generator(gen_lst, batch_size)

        G_whole = netD2(fake)
        G_whole = G_whole.mean()

        # Update gradients.
        G_whole.backward(neg_one)
        optimizerG.step()
        optimizerG.zero_grad()
        optimizerRNN.step()
        optimizerRNN.zero_grad()

        G_whole_cost = -G_whole

        if cuda:
            G_whole_cost = G_whole_cost.cpu()
        G_whole_epoch.append(G_whole_cost.data.numpy()[0])

        ################
        # Compute valid
        ################
        """
        Compute valid only once
        """
        # Noise
        noise_valid = random_noise(latent_dim, same_z, one_z, cuda)

        valid_real_data = next(valid_iter)['X']

        # ======== Compute valid D1 ========
        netD1.zero_grad()

        valid_real_piece_Var = numpy_to_var_piece(valid_real_data, cuda)
        D1_real_valid = netD1(valid_real_piece_Var)
        D1_real_valid = D1_real_valid.mean()  # avg loss

        # b) compute loss contribution from generated data, then backprop.
        gen_lst, hidden = pass_through_rnn_g(noise_valid, hidden, same_z, one_z, rnnModel, netG, batch_size)
        fake_valid = piece_fake_generator(gen_lst, batch_size)
        D1_fake_valid = netD1(fake_valid.detach())
        D1_fake_valid = D1_fake_valid.mean()

        # c) compute gradient penalty and backprop
        gradient_penalty_valid1 = calc_gradient_penalty(netD1, valid_real_piece_Var.data,
                                                        fake_valid.data, batch_size, lmbda,
                                                        use_cuda=cuda)
        # Compute metrics and record in batch history.
        D1_cost_valid = D1_real_valid - D1_fake_valid + gradient_penalty_valid1
        D1_wass_valid = D1_real_valid - D1_fake_valid

        if cuda:
            D1_cost_valid = D1_cost_valid.cpu()
            D1_wass_valid = D1_wass_valid.cpu()

        # Record costs
        D1_cost_valid_epoch.append(D1_cost_valid.data.numpy()[0])
        D1_wass_valid_epoch.append(D1_wass_valid.data.numpy()[0])

        # ======== Compute valid D2 ========
        netD2.zero_grad()

        valid_real_Var = numpy_to_var(valid_real_data, cuda)
        D2_real_valid = netD2(valid_real_Var)
        D2_real_valid = D2_real_valid.mean()  # avg loss

        # b) compute loss contribution from generated data, then backprop.
        gen_lst, hidden = pass_through_rnn_g(noise_valid, hidden, same_z, one_z, rnnModel, netG, batch_size)
        fake_valid = batch_fake_generator(gen_lst, batch_size)
        D2_fake_valid = netD2(fake_valid.detach())
        D2_fake_valid = D2_fake_valid.mean()

        # c) compute gradient penalty and backprop
        gradient_penalty_valid2 = calc_gradient_penalty(netD2, valid_real_Var.data,
                                                        fake_valid.data, batch_size, lmbda,
                                                        use_cuda=cuda)
        # Compute metrics and record in batch history.
        D2_cost_valid = D2_real_valid - D2_fake_valid + gradient_penalty_valid2
        D2_wass_valid = D2_real_valid - D2_fake_valid

        if cuda:
            D2_cost_valid = D2_cost_valid.cpu()
            D2_wass_valid = D2_wass_valid.cpu()

        # Record costs
        D2_cost_valid_epoch.append(D2_cost_valid.data.numpy()[0])
        D2_wass_valid_epoch.append(D2_wass_valid.data.numpy()[0])

        """
        Here detach the hidden to disconnect the graph between batches.
        See https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time-while-using-custom-loss-function/12360/2
        """
        hidden = hidden.detach()

        # (BATCH_NUM // 5)
        LOG_PER = (BATCH_NUM // 5) if MAX_DATA_NUM > 1000 else 1
        if i % LOG_PER == 0:
            LOGGER.info(
                "{} Epoch={} Batch: {}/{} D1_w:{:.4f} | D2_w:{:.4f} | G:{:.4f} | G_whole_cost:{:.4f}".format(time_since(start), epoch,
                                                                                     i, BATCH_NUM,
                                                                                       D1_wass_train.data.numpy()[0],
                                                                                       D2_wass_train.data.numpy()[0],
                                                                                     G_cost.data.numpy()[0],
                                                                                     G_whole_cost.data.numpy()[0]))

        gc.collect()

    # Save the average cost of batches in every epoch.
    D1_cost_train_epoch_avg = sum(D1_cost_train_epoch) / float(len(D1_cost_train_epoch))
    D1_wass_train_epoch_avg = sum(D1_wass_train_epoch) / float(len(D1_wass_train_epoch))
    D1_cost_valid_epoch_avg = sum(D1_cost_valid_epoch) / float(len(D1_cost_valid_epoch))
    D1_wass_valid_epoch_avg = sum(D1_wass_valid_epoch) / float(len(D1_wass_valid_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D1_costs_train.append(D1_cost_train_epoch_avg)
    D1_wasses_train.append(D1_wass_train_epoch_avg)
    D1_costs_valid.append(D1_cost_valid_epoch_avg)
    D1_wasses_valid.append(D1_wass_valid_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    D2_cost_train_epoch_avg = sum(D2_cost_train_epoch) / float(len(D2_cost_train_epoch))
    D2_wass_train_epoch_avg = sum(D2_wass_train_epoch) / float(len(D2_wass_train_epoch))
    D2_cost_valid_epoch_avg = sum(D2_cost_valid_epoch) / float(len(D2_cost_valid_epoch))
    D2_wass_valid_epoch_avg = sum(D2_wass_valid_epoch) / float(len(D2_wass_valid_epoch))
    G_whole_epoch_avg = sum(G_whole_epoch) / float(len(G_whole_epoch))

    D2_costs_train.append(D2_cost_train_epoch_avg)
    D2_wasses_train.append(D2_wass_train_epoch_avg)
    D2_costs_valid.append(D2_cost_valid_epoch_avg)
    D2_wasses_valid.append(D2_wass_valid_epoch_avg)
    G_whole_costs.append(G_whole_epoch_avg)

    LOGGER.info("{} D1_wass_train:{:.4f} | D1_wass_valid:{:.4f} | D2_wass_train:{:.4f} | D2_wass_valid:{:.4f} "
                "| G_cost:{:.4f} | G_whole:{:.4f}".format(time_since(start),
                                         D1_wass_train_epoch_avg,
                                         D1_wass_valid_epoch_avg,
                                         D2_wass_train_epoch_avg,
                                         D2_wass_valid_epoch_avg,
                                         G_cost_epoch_avg,
                                         G_whole_epoch_avg))

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

LOGGER.info('>>>>>>>Training finished !<<<<<<<')


save_model(EPOCHS, output_dir, netD1, netD2, netG, rnnModel)
plot_loss_wass(D1_wasses_train, D1_wasses_valid, G_costs, output_dir, "g1d1")
plot_loss_wass(D2_wasses_train, D2_wasses_valid, G_whole_costs, output_dir, "rnnd2")

LOGGER.info("All finished!")



