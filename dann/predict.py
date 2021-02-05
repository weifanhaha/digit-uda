import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from torch import optim
from torch.utils.data import DataLoader

from models import DANN
from reproduce_dataset import ReproduceDataset


def main(args):
    # read from args
    test_path = args.test_path
    d_target = args.d_target
    output_predict_path = args.output_predict_path

    ########## Arguments ##########
    batch_size = 128

    # svhn, usps, mnistm
    if d_target == "mnistm":
        d_source = "usps"
    elif d_target == "svhn":
        d_source = "mnistm"
    else:
        d_source = "svhn"

    output_model_path = "./hw3-3/models/{}_{}_domain.pth".format(
        d_source, d_target)

    #############################

    dataset = ReproduceDataset(test_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DANN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(
        output_model_path, map_location=device))

    model.eval()
    preds = []

    for idx, targets in enumerate(dataloader):
        target_images = targets.to(device)
        target_bs = target_images.shape[0]

        with torch.no_grad():
            label_output, domain_output = model(target_images, 1.0)

        # calculate label acc
        _, pred_labels = torch.max(label_output, 1)
        preds.append(pred_labels)

    # save to predict
    pred = torch.cat(preds).cpu().numpy()
    image_names = ['{:05}.png'.format(i) for i in range(len(pred))]

    pd.DataFrame({
        'image_name': image_names,
        'label': pred
    }).to_csv(output_predict_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--test_path", help="path of the test image directory",
                        dest="test_path", default="../hw3_data/digits/mnistm/test")

    parser.add_argument("--d_target", help="target dataset",
                        dest="d_target", default="mnistm")

    parser.add_argument("--output_predict_path", help="path of the output csv",
                        dest="output_predict_path", default="./prediction.csv")

    args = parser.parse_args()
    main(args)
