import torch


def TransformONNX():
    model = torch.load("checkpoints/best.pth")
    model.eval()

    x = torch.rand(10, 15)  # 生成张量
    x = x.long()
    export_onnx_file = "test.onnx"  # 目的ONNX文件名
    torch.onnx.export(model, x, export_onnx_file, verbose=True)
