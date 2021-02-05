import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from torch import optim
from torch.utils.data import DataLoader

from models import LeNetEncoder, LeNetClassifier
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

    output_src_classifier_path = "./hw3-4/models/src_classifier_{}_{}.pth".format(
        d_source, d_target)
    output_tgt_encoder_path = "./hw3-4//models/tgt_encoder_{}_{}.pth".format(
        d_source, d_target)

    #############################

    dataset = ReproduceDataset(test_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init models
    tgt_encoder = LeNetEncoder()
    src_classifier = LeNetClassifier()

    # to device
    tgt_encoder.to(device)
    src_classifier.to(device)

    # init weights
    tgt_encoder.load_state_dict(torch.load(
        output_tgt_encoder_path, map_location=device))

    src_classifier.load_state_dict(torch.load(
        output_src_classifier_path, map_location=device))

    tgt_encoder.eval()
    src_classifier.eval()

    all_pred = []

    for idx, targets in enumerate(dataloader):
        target_images = targets.to(device)
        target_bs = target_images.shape[0]

        with torch.no_grad():
            preds = src_classifier(tgt_encoder(target_images))

        # calculate label acc
        _, pred_labels = torch.max(preds, 1)
        all_pred.append(pred_labels)

    # save to predict
    pred = torch.cat(all_pred).cpu().numpy()
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
