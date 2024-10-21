import training
import utils


# Main training loop MNIST
if __name__ == "__main__":

    # create and parse arguments
    args = utils.parse_args()

    # setup logging
    utils.setup_logging()

    # load config
    config = utils.load_config(args.config)
    # config = utils.modify_config(config, args.params)

    # get device
    device = utils.get_device()

    # load model
    model = utils.get_model(config['model'])

    # unsupervised training (if using a Hebbian learning rule)
    if config['model']['type'] == 'GenHebb':
        unsup_trainloader = utils.load_data(config['training'], 'unsupervised', device)
        training.unsupervised(
            model,
            unsup_trainloader,
            config['training']['unsupervised']['epochs'],
            config['training']['unsupervised']['lr'],
            device
        )

    # supervised training
    sup_trainloader, sup_testloader = utils.load_data(config['training'], 'supervised', device)
    training.supervised(
        model,
        sup_trainloader,
        sup_testloader,
        config['training']['supervised']['epochs'],
        config['training']['supervised']['lr'],
        device
    )

    # save model if specified
    if args.save:
        utils.save_model(model)