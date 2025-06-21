import argparse

from commons.utils import set_seed
from models.utils import build_model
from dataset.utils import build_dataloader_imagenet
from algorithms.conf_scalings import ConformalTemperatureScaling, ConformalPlattScaling, ConformalVectorScaling
from algorithms.scalings import Identity, TemperatureScaling, PlattScaling, VectorScaling
from algorithms.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description='ConfTS')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--trials', type=int, default=1, help='number of trials')
    parser.add_argument('--model', type=str, default='resnet50', help='model')
    parser.add_argument('--data_dir', '-s', type=str, default='/mnt/sharedata/ssd3/common/datasets',
                        help='dataset name.')
    parser.add_argument('--conformal', type=str, default='aps', help='conformal prediction')
    parser.add_argument('--alpha', type=float, default=0.1, help="error rate")
    parser.add_argument('--cal_num', type=int, default=10000, help="calibration size")
    parser.add_argument('--conf_num', type=int, default=5000, help="conformal size")
    parser.add_argument('--temp_num', type=int, default=5000, help="temperature size")
    parser.add_argument('--gamma', type=float, default=0.5, help="conformal size")
    parser.add_argument('--preprocess', type=str, default="confts", help="conformal size")
    parser.add_argument('--penalty', type=float, default=0.001, help="conformal size")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    set_seed(args.seed)
    trails = args.trials
    model_name = args.model
    data_dir = args.data_dir
    conformal = args.conformal
    alpha = args.alpha
    cal_num = args.cal_num
    conf_num = args.conf_num
    temp_num = args.temp_num
    gamma = args.gamma
    pre = args.preprocess
    penalty = args.penalty

    model = build_model(model_name)
    model = model.cuda()
    calib_calibloader, conf_calibloader, testloader = build_dataloader_imagenet(data_dir, conf_num=conf_num,
                                                                                temp_num=temp_num)
    if pre == "confts":
        preprocessor = ConformalTemperatureScaling(model, alpha)
    elif pre == "confps":
        preprocessor = ConformalPlattScaling(model, alpha)
    elif pre == "confvs":
        preprocessor = ConformalVectorScaling(model, alpha)
    elif pre == "ts":
        preprocessor = TemperatureScaling()
    elif pre == "ps":
        preprocessor = PlattScaling()
    elif pre == "vs":
        preprocessor = VectorScaling()
    else:
        preprocessor = Identity()

    predictor = Predictor(model, preprocessor, conformal, alpha, random=True, penalty=penalty)
    ece_before, ece_after = predictor.calibrate(calib_calibloader, conf_calibloader)
    result = predictor.evaluate(testloader)
    print(result)
