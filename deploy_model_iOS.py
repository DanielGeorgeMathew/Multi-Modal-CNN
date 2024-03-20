import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import coremltools as ct
from src.models.model import SegNet

# This script can be used to save the MLModel file which can be directly used to perform inference on mobile IoS device.


class WrappedSegnet(nn.Module):
    def __init__(self):
        super(WrappedSegnet, self).__init__()
        self.model = SegNet(
            height=640,
            width=480,
            num_classes=1,
            encoder_rgb='resnet34',
            encoder_depth='resnet34',
            encoder_block='NonBottleneck1D',
            activation='relu',
            encoder_decoder_fusion='add',
            context_module='ppm',
            nr_decoder_blocks=[3] * 3,
            channels_decoder=[512, 256, 128],
            fuse_depth_in_rgb_encoder='SE-add',
            upsampling='learned-3x3-zeropad'
        )
        model.load_state_dict(torch.load(
            'PATH TO MODEL WEIGHTS',
            map_location=torch.device('cpu'))['model'])


        self.model.backbone.qconfiq = None
        self.model.eval()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, rgb, depth):
        rgb = self.quant(rgb)
        depth = self.quant(depth)
        res = self.model(rgb, depth)[:, 0, :, :]
        res = self.dequant(res)
        return res


num_calibration_batches = 6



example_input = (torch.rand(1, 3, 640, 480), torch.rand(1, 1, 640, 480))


model = WrappedSegnet()

# QUANTIZING THE MODEL
# model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# SCRIPTING/TRACING THE MODEL
traced_model = torch.jit.trace(model, example_input)
scripted_model = torch.jit.script(model)

# MODEL OPTIMIZATION
# optimized_model = optimize_for_mobile((scripted_model), backend='Metal')

# SAVING MODEL
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name='image_input', shape=example_input[0].shape, scale=1 / 255.0, bias=0, channel_first=True),
            ct.ImageType(name='depth_input', shape=example_input[1].shape, scale=1 / 65535.0, bias=0, channel_first=True)]
)

mlmodel.save(".//Wound_segmentation_quantized.mlmodel")