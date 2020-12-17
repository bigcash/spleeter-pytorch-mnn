import torch
import torchvision
from spleeter.util import tf2pytorch
from spleeter.unet import UNet


def load_ckpt(model, ckpt):
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            target_shape = state_dict[k].shape
            assert target_shape == v.shape
            state_dict.update({k: torch.from_numpy(v)})
        else:
            print('Ignore ', k)

    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    checkpoint_path = "pretrained/2stems/model"
    num_instrumments = 2
    ckpts = tf2pytorch(checkpoint_path, num_instrumments)
    net = UNet(2)
    ckpt = ckpts[0]
    net = load_ckpt(net, ckpt)
    dummy_input = torch.randn(1, 2, 512, 1024, device='cpu')
    torch.onnx.export(net, dummy_input, "vocals.onnx", verbose=True, export_params=True)
    ckpt = ckpts[1]
    net = load_ckpt(net, ckpt)
    dummy_input = torch.randn(1, 2, 512, 1024, device='cpu')
    torch.onnx.export(net, dummy_input, "accompaniment.onnx", verbose=True, export_params=True)
