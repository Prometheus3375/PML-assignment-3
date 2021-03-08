if __name__ == '__main__':
    default_model = 'model.pth'

    from argparse import ArgumentParser

    parser = ArgumentParser(
        usage='Runs model on test.ru.txt and save results to answer.zip with answer.txt',
    )
    parser.add_argument(
        'model',
        help=f'path to weights of the model. Default: {default_model!r}',
        default=default_model,
    )

    args = parser.parse_args()

    import torch

    weights = torch.load(args.model)
    # TODO, https://stackoverflow.com/questions/42214376/zip-single-file
