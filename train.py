from __future__ import print_function
import torch
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from torch.autograd import Variable
import os
import numpy as np

from dataset import train_loader, val_loader


def export_quant(model, name, device):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d}, dtype=torch.qint8)
    onnx_file_path = name + " - quantized.onnx"
    torch.onnx.export(quantized_model, dummy_input, onnx_file_path, verbose=True, input_names=['input'],
                      output_names=['output'])
    print("Модель успешно экспортирована в формат ONNX и сохранена в", onnx_file_path)


from unet_model import UNet

model = UNet()
model.cuda()


def custom_loss_function(output, target):
    di = target - output
    n = (224 * 224)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    return loss.mean()


# default SGD optimiser - don't work
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

dtype = torch.cuda.FloatTensor


def plot_n_save_fig(epoch, plot_input, output, actual_output, path):
    F = plt.figure(1, (30, 60))
    F.subplots_adjust(left=0.05, right=0.95)
    plot_grid(F, plot_input, output, actual_output, 1)
    plt.savefig("plots/" + path + "_" + str(epoch) + ".jpg")
    plt.show()


def plot_grid(fig, plot_input, output, actual_output, row_no):
    grid = ImageGrid(fig, 141, nrows_ncols=(row_no, 4), axes_pad=0.05, label_mode="1")
    for i in range(row_no):
        for j in range(3):
            if (j == 0):
                grid[i * 4 + j].imshow(np.transpose(plot_input[i], (1, 2, 0)), interpolation="nearest")
            if (j == 1):
                grid[i * 4 + j].imshow(np.transpose(output[i][0].detach().cpu().numpy(), (0, 1)),
                                       interpolation="nearest")
            if (j == 2):
                grid[i * 4 + j].imshow(np.transpose(actual_output[i][0].detach().cpu().numpy(), (0, 1)),
                                       interpolation="nearest")


# All Error Function
def threeshold_percentage(output, target, threeshold_val):
    d1 = torch.exp(output) / torch.exp(target)
    d2 = torch.exp(target) / torch.exp(output)
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1, 2, 3))
    threeshold_mat = count_mat / (output.shape[2] * output.shape[3])
    return threeshold_mat.mean()


def rmse_linear(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target):
    diff = output - target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return mse.mean()


def abs_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()


def squared_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return square_relative_diff.mean()


#############################################
log_interval = 1


def mean_absolute_error(output, target):
    absolute_diff = torch.abs(output - target)
    mae = torch.mean(absolute_diff)
    return mae.item()


def train_Unet(epoch):
    model.train()
    train_coarse_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        x_rgb = sample['rgb'].cuda()
        y_depth = sample['depth'].cuda()

        optimizer.zero_grad()
        y_hat = model(x_rgb.type(dtype))
        loss = custom_loss_function(y_hat, y_depth)
        loss.backward()
        optimizer.step()
        train_coarse_loss += loss.item()
        if epoch % log_interval == 0:
            training_tag = "training loss epoch:" + str(epoch)
            # logger.scalar_summary(training_tag, loss.item(), batch_idx)
    train_coarse_loss /= (batch_idx + 1)
    return train_coarse_loss


print(
    "Epochs:     Train_loss  Val_loss       rmse_lin    rmse_log    abs_rel.  square_relative")
print(
    "Paper Val:                               (0.871)     (0.283)     (0.228)     (0.223)")


def validate_Unet(epoch, training_loss):
    model.eval()
    validation_loss = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    mean_absolute_error_loss = 0  # Initialize MAE loss
    with torch.no_grad():
        for batch_idx, sample in enumerate(train_loader):
            x_rgb = sample['rgb'].cuda()
            y_depth = sample['depth'].cuda()

            y_hat = model(x_rgb.type(dtype))
            loss = custom_loss_function(y_hat, y_depth)
            validation_loss += loss
            # all error functions
            rmse_linear_loss += rmse_linear(y_hat, y_depth)
            rmse_log_loss += rmse_log(y_hat, y_depth)
            abs_relative_difference_loss += abs_relative_difference(y_hat, y_depth)
            squared_relative_difference_loss += squared_relative_difference(y_hat, y_depth)
        validation_loss /= (batch_idx + 1)
        rmse_linear_loss /= (batch_idx + 1)
        rmse_log_loss /= (batch_idx + 1)
        abs_relative_difference_loss /= (batch_idx + 1)
        squared_relative_difference_loss /= (batch_idx + 1)
        mean_absolute_error_loss /= (batch_idx + 1)
        print(
            'Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.
            format(epoch, training_loss, validation_loss, rmse_linear_loss, rmse_log_loss, mean_absolute_error_loss))


folder_name = "models/"
if not os.path.exists(folder_name): os.mkdir(folder_name)
epochs = 10
print("********* Training the Unet Model **************")
for epoch in range(1, epochs + 1):
    training_loss = train_Unet(epoch)
    if epoch % 1 == 0:
        validate_Unet(epoch, training_loss)
    if epoch % 10 == 0:
        export_quant(model, "models/Unet", torch.device("cuda"))
