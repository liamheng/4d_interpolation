import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from procedures import preprocedure
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils import logger
from utils.visualizer import Visualizer

if __name__ == '__main__':
    # get and modify the options
    train_options = TrainOptions()
    opt = train_options.opt
    preprocedure(opt)
    train_options.print_options(opt)

    logger.info('Current training name: ' + opt.name)
    logger.info('Current training secondary name: ' + opt.secondary_dirname)

    # create dataset and dataloader
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    logger.info('The number of training batches = %d' % dataset_size)

    # --- LR warmup support (iteration-based) ---
    # warmup_percentage in [0, 1]; 0 disables warmup
    warmup_pct = float(getattr(opt, 'warmup_percentage', 0.0) or 0.0)
    # sample_repeat affects iteration count per epoch
    opt.iters_per_epoch = int(dataset_size * getattr(opt, 'sample_repeat', 1))
    opt.total_iterations = int(getattr(opt, 'epochs_num', 0) * opt.iters_per_epoch)
    if warmup_pct > 0.0:
        opt.warmup_iters = int(opt.total_iterations * warmup_pct)
        if opt.warmup_iters < 1 and opt.total_iterations > 0:
            opt.warmup_iters = 1
    else:
        opt.warmup_iters = 0

    # create model
    model = create_model(opt)
    model.setup()

    if not opt.no_visdom:
        visualizer = Visualizer(opt)
        visualizer.reset()

    total_iters = opt.total_iterations  # total number of iterations
    logger.info('The total number of training iterations = %d' % total_iters)

    opt.current_iteration = 0  # the total number of training iterations

    for epoch in range(opt.epochs_num):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        opt.current_epoch = epoch

        if opt.display_id > 0 and not opt.no_visdom:
            visualizer.reset()

        for _ in range(opt.sample_repeat):
            for i, data in enumerate(dataset):
                iter_start_time = time.time()  # timer for computation per iteration
                if opt.current_iteration % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                opt.current_iteration += 1
                epoch_iter += opt.batch_size

                # unpack data from dataset
                model.set_input(data)
                # calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

                # Iteration-based LR warmup: update scheduler per-iteration only during warmup phase
                if getattr(opt, 'warmup_iters', 0) > 0 and opt.current_iteration <= opt.warmup_iters:
                    opt.lr_update_mode = 'iter'
                    model.update_learning_rate()
                    opt.lr_update_mode = 'epoch'

                # display images on visdom and save images to an HTML file
                if opt.current_iteration % opt.display_freq == 0 and not opt.no_visdom:
                    save_result = opt.current_iteration % opt.update_html_freq == 0
                    # in train mode, compute_visuals() will not be called automatically
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(),
                                                       epoch, save_result)

                # print training losses and save logging information to the disk
                if opt.current_iteration % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    message = 'epoch: %d, iteration: %d, computation time: %.3f ms, data loading time: %.3f ms, ' % (
                        epoch, epoch_iter, 1000 * t_comp, 1000 * t_data)
                    for k, v in losses.items():
                        message += '%s: %.3f ' % (k, v)
                    logger.info(message)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / (dataset_size * opt.sample_repeat),
                                                       losses)

                iter_data_time = time.time()

        # update learning rates at the end of every epoch
        opt.lr_update_mode = 'epoch'
        model.update_learning_rate()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, opt.current_iteration))
            model.save_networks('last')
            model.save_networks(epoch)

        logger.info(
            'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epochs_num, time.time() - epoch_start_time))

        if epoch >= opt.stop_epoch:
            logger.info('Reached stop_epoch %d. Stopping training.' % opt.stop_epoch)
            break

    logger.info('Training process finishes')
    model.save_networks('last')
