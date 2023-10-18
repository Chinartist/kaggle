import argparse
import torch_tensorrt
import tensorrt
import torch
import torch.nn as nn
from model import NextViT

parser = argparse.ArgumentParser()

parser.add_argument("--snapshot")
parser.add_argument("--output")
parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--half", type=bool, default=True)

args = parser.parse_args()

print(args)

model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, num_classes=1)
model.proj_head = nn.Linear(1024, 1)

f = torch.load(args.snapshot, map_location=lambda storage, loc: storage)
model.load_state_dict(f, strict=False)

if args.half:
    model.half()
    dtype = torch.half
else:
    dtype = torch.float32

model.eval().cuda()
model.merge_bn()

# The compiled module will have precision as specified by "op_precision".
# import torch_tensorrt
print('start compiling')
trt_model_fp16 = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
        [args.batch_size, 3, args.height, args.width],
        dtype=dtype
    )],
    enabled_precisions={dtype},  # Run with FP16
    workspace_size=1 << 32,
    require_full_compilation=True,
)

print('finished compiling')
torch.jit.save(trt_model_fp16, args.output)
print(f'model saved: {args.output}')
